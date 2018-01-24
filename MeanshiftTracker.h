//
// Created by Patrick Werner (boonto) on 25.11.17.
//

#ifndef CV_PROJECT_MEANSHIFTTRACKER_H
#define CV_PROJECT_MEANSHIFTTRACKER_H

#include <opencv2/tracking.hpp>
#include "Tracker.h"

class MeanshiftTracker : public Tracker {
public:
    struct Parameters {
        int nMaxIterations = 200;
        int nBins = 32;
    };

    explicit MeanshiftTracker(const Parameters &parameters) :
            parameters(parameters),
            initialized(false),
            targetHist() {
    }

    void track(const cv::Mat &image, cv::Rect2f &roi) override;

    void reset() override {
        initialized = false;
    }

    float evaluate(const cv::Rect2f &roi, const cv::Rect2f &groundTruthRoi) const override;

    std::string classname() const override {
        return "MeanshiftTracker";
    }

private:
    Parameters parameters;
    bool initialized;
    cv::Mat targetHist;

    void initialize(const cv::Mat &image, cv::Rect2f &roi);

    cv::Mat getHistogram(const cv::Mat &image, int nBins) const;

    cv::Mat getBackProject(const cv::Mat &image, const cv::Mat &hist) const;

    void roiToBounds(cv::Rect2f &roi, const cv::Size size) const {
        // Ensure roi is in image bounds
        roi.width = std::min(static_cast<float>(size.width), roi.width);
        roi.height = std::min(static_cast<float>(size.height), roi.height);
        roi.x = std::min(std::max(0.0f, roi.x), size.width - roi.width);
        roi.y = std::min(std::max(0.0f, roi.y), size.height - roi.height);
    }
};


#endif //CV_PROJECT_MEANSHIFTTRACKER_H
