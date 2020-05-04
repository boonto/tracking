//
// Created by Patrick Werner (boonto) on 25.11.17.
//

#ifndef TRACKING_TRACKER_H
#define TRACKING_TRACKER_H

#include <opencv2/core/types.hpp>

class Tracker {
public:
    virtual ~Tracker() = default;

    virtual void track(const cv::Mat &image, cv::Rect2f &roi) = 0;

    virtual void reset() = 0;

    virtual float evaluate(const cv::Rect2f &roi, const cv::Rect2f &groundTruthRoi) const = 0;

    virtual std::string classname() const = 0;
};

#endif //TRACKING_TRACKER_H
