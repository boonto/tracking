//
// Created by Patrick Werner (boonto) on 25.11.17.
//

#include "MeanshiftTracker.h"

void MeanshiftTracker::track(const cv::Mat &image, cv::Rect2f &roi) {
    roiToBounds(roi, image.size());

    if (!initialized) {
        initialize(image, roi);
    }

    auto currBackBGR = getBackProject(image, targetHist);

    for (auto i = 0; i < parameters.nMaxIterations; ++i) {
        // Calculate center of mass according to OpenCV doc
        auto moments = cv::moments(currBackBGR(roi));
        auto centroid = cv::Point2f(static_cast<float>(moments.m10 / moments.m00),
                                    static_cast<float>(moments.m01 / moments.m00));

        // Calculate update
        auto dX = centroid.x - roi.width * 0.5f;
        auto dY = centroid.y - roi.height * 0.5f;

        // Update roi
        roi.x += dX;
        roi.y += dY;

        roiToBounds(roi, image.size());

        if (cv::norm(cv::Point2f(dX, dY)) < 0.2f) {
            break;
        }
    }
}

void MeanshiftTracker::initialize(const cv::Mat &image, cv::Rect2f &roi) {
    auto win = image(roi).clone();
    targetHist = getHistogram(win, parameters.nBins);
    initialized = true;
}

cv::Mat MeanshiftTracker::getHistogram(const cv::Mat &image, int nBins) const {
    int histSizes[] = {nBins, nBins, nBins};

    float range[] = {0, 255};
    const float *histRanges[] = {range, range, range};

    int channels[] = {0, 1, 2};
    auto hist = cv::Mat();
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 3, histSizes, histRanges);
    cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    return hist;
}

cv::Mat MeanshiftTracker::getBackProject(const cv::Mat &image, const cv::Mat &hist) const {
    float range[] = {0, 255};
    const float *histRanges[] = {range, range, range};

    int channels[] = {0, 1, 2};
    auto back = cv::Mat();
    cv::calcBackProject(&image, 1, channels, hist, back, histRanges);
    back.convertTo(back, CV_32F);
    return back;
}

float MeanshiftTracker::evaluate(const cv::Rect2f &roi, const cv::Rect2f &groundTruthRoi) const {
    // Intersection over union
    return (roi & groundTruthRoi).area() / (roi | groundTruthRoi).area();
}
