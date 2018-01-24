//
// Created by Patrick Werner (boonto) on 25.11.17.
//

#include "LucasKanadeTracker.h"
#include <opencv2/highgui.hpp>

void LucasKanadeTracker::track(const cv::Mat &image, cv::Rect2f &roi) {
    if (!initialized) {
        initialize(image, roi);
        return;
    }

    auto currentImage = prepareImage(image);

    auto w = static_cast<int>(std::floor(parameters.windowSize / 2.0f));

    // Compute the x and y derivatives for the whole image
    auto derivatives = computeDerivatives(prevImage);

    for (auto &feature : features) {
        auto window = buildWindow(feature, w);

        // Next feature if window is too small
        if (window.size().width < 2 || window.size().height < 2) continue;

        // Cut out the window from the derivatives
        auto derivativeXWindow = std::get<0>(derivatives)(window).clone();
        auto derivativeYWindow = std::get<1>(derivatives)(window).clone();
        // Cut out the window of the previous frame
        auto prevWindow = prevImage(window).clone();

        // Iteratively figure out new feature position
        auto prevX = 0.0f;
        auto prevY = 0.0f;
        for (auto i = 0; i < parameters.nMaxIterations; ++i) {
            // Build new window
            window = buildWindow(feature, w);

            if (window.size().width < 1 || window.size().height < 1) continue;

            // Cut out the window of the current frame
            auto currWindow = currentImage(window).clone();

            // Get time derivative
            auto derivativeTWindow = cv::Mat();
            cv::resize(currWindow, derivativeTWindow, prevWindow.size());
            derivativeTWindow = cv::Mat(derivativeTWindow - prevWindow);

            // Rearrange matrices
            auto A1 = cv::Mat(derivativeXWindow.reshape(0, 1).t());
            auto A2 = cv::Mat(derivativeYWindow.reshape(0, 1).t());
            auto b = cv::Mat(-derivativeTWindow.reshape(0, 1).t());

            if (parameters.bUseGauss) {
                filter(A1, A2, b, derivativeXWindow.size().width);
            }

            // Combine A1 and A2
            auto A = cv::Mat();
            cv::hconcat(A1, A2, A);

            // Solve the over determined equation system
            // All methods are identical
//            auto v = cv::Mat();
//            cv::solve(A, b, v, cv::DECOMP_SVD);
            auto v = cv::Mat(A.inv(cv::DECOMP_SVD) * b);
//            auto v = cv::Mat((A.t() * A).inv() * A.t() * b);

            // Update the feature position
            feature.x += v.at<float>(0);
            feature.y += v.at<float>(1);

            // Stop the loop if the changes are too small
            if (std::abs(prevX - feature.x) < parameters.iterationEps &&
                std::abs(prevY - feature.y) < parameters.iterationEps) {
                break;
            }
            prevX = feature.x;
            prevY = feature.y;
        }
    }

    // Copy current to previous mat
    currentImage.copyTo(prevImage);

    roi = updateRoi();
}

void LucasKanadeTracker::initialize(const cv::Mat &image, const cv::Rect2f &roi) {
    // Prepare image for tracking
    auto gray = cv::Mat();
    cv::cvtColor(image, gray, CV_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    gray.copyTo(prevImage);

    // Get new tracking points
    auto mask = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(0));
    if (!roi.empty()) {
        mask(roi).setTo(cv::Scalar(255));
    } else {
        mask.setTo(cv::Scalar(255));
    }
    cv::goodFeaturesToTrack(gray, features, parameters.nFeatures, parameters.qualityLevel, parameters.minDistance,
                            mask);

    if (!features.empty()) {
//        cv::cornerSubPix(gray, features, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria());
        initialized = true;
        nInitialPoints = static_cast<int>(features.size());
    }
}

cv::Rect2f LucasKanadeTracker::updateRoi() const {
    auto minX = static_cast<float>(prevImage.size().width);
    auto minY = static_cast<float>(prevImage.size().height);
    auto maxX = 0.0f;
    auto maxY = 0.0f;

    for (auto const &feature : features) {
        minX = std::min(minX, feature.x);
        minY = std::min(minY, feature.y);
        maxX = std::max(maxX, feature.x);
        maxY = std::max(maxY, feature.y);
    }
    return cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
}

cv::Mat LucasKanadeTracker::prepareImage(const cv::Mat &inputImage) const {
    auto outputImage = cv::Mat();
    // Prepare frame for tracking
    cv::cvtColor(inputImage, outputImage, CV_BGR2GRAY);
    outputImage.convertTo(outputImage, CV_32F);
    return outputImage;
}

std::tuple<cv::Mat, cv::Mat> LucasKanadeTracker::computeDerivatives(const cv::Mat &image) const {
    auto derivativeX = cv::Mat();
    auto derivativeY = cv::Mat();

    //    cv::Sobel(prevImage, derivativeX, -1, 1, 0);
    //    cv::Sobel(prevImage, derivativeY, -1, 0, 1);

    cv::Scharr(image, derivativeX, -1, 1, 0);
    cv::Scharr(image, derivativeY, -1, 0, 1);
    derivativeX *= 0.25f;
    derivativeY *= 0.25f;

    return std::make_tuple(derivativeX, derivativeY);
}

cv::Rect2f LucasKanadeTracker::buildWindow(const cv::Point2f &feature, int w) const {
    // Build window out of feature
    auto left = std::floor(std::max(0.0f, feature.x - w));
    auto top = std::floor(std::max(0.0f, feature.y - w));
    auto right = std::ceil(std::min(static_cast<float>(prevImage.size().width), feature.x + w));
    auto bottom = std::ceil(std::min(static_cast<float>(prevImage.size().height), feature.y + w));

    return cv::Rect2f(left, top, right - left, bottom - top);
}

void LucasKanadeTracker::filter(cv::Mat &A1, cv::Mat &A2, cv::Mat &b, int w) const {
    auto gauss = cv::getGaussianKernel(w, parameters.gaussSigma, CV_32F);
    gauss = gauss * gauss.t();
    cv::resize(gauss, gauss, A1.size());
    gauss = gauss.reshape(0, 1).t();
    A1 = A1.mul(gauss);
    A2 = A2.mul(gauss);
    b = b.mul(gauss);
}

float LucasKanadeTracker::evaluate(const cv::Rect2f &roi, const cv::Rect2f &groundTruthRoi) const {
    int nInside = 0;
    for (auto const &feature : features) {
        nInside += (groundTruthRoi.contains(feature)) ? 1 : 0;
    }

    return nInside / static_cast<float>(nInitialPoints);
}
