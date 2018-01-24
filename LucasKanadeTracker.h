//
// Created by Patrick Werner (boonto) on 25.11.17.
//

#ifndef CV_PROJECT_LUCASKANADETRACKER_H
#define CV_PROJECT_LUCASKANADETRACKER_H

#include <opencv2/tracking.hpp>
#include "Tracker.h"

class LucasKanadeTracker : public Tracker {
public:
    struct Parameters {
        int nFeatures = 30;
        double qualityLevel = 0.15;
        double minDistance = 5;
        bool bUseGauss = false;
        float gaussSigma = 2.0f;
        int nMaxIterations = 40;
        int windowSize = 21;
        float iterationEps = 0.05f;
    };

    explicit LucasKanadeTracker(const Parameters &parameters) :
            parameters(parameters),
            initialized(false),
            features(),
            prevImage(),
            nInitialPoints(0) {
    }

    void track(const cv::Mat &image, cv::Rect2f &roi) override;

    void reset() override {
        initialized = false;
    }

    void display(cv::Mat &display) const {
        // Show feature points
        for (const auto &corner : features) {
            cv::circle(display, corner, 3, cv::Scalar(0, 255, 0), -1);
        }
    }

    float evaluate(const cv::Rect2f &roi, const cv::Rect2f &groundTruthRoi) const override;

    std::string classname() const override {
        return "LucasKanadeTracker";
    }

private:
    Parameters parameters;
    bool initialized;
    std::vector<cv::Point2f> features;
    cv::Mat prevImage;
    int nInitialPoints;

    void initialize(const cv::Mat &image, const cv::Rect2f &roi);

    cv::Rect2f updateRoi() const;

    cv::Mat prepareImage(const cv::Mat &inputImage) const;

    std::tuple<cv::Mat, cv::Mat> computeDerivatives(const cv::Mat &image) const;

    cv::Rect2f buildWindow(const cv::Point2f &feature, int w) const;

    void filter(cv::Mat &A1, cv::Mat &A2, cv::Mat &b, int i) const;
};


#endif //CV_PROJECT_LUCASKANADETRACKER_H
