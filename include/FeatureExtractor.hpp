
#pragma once

#include "ha_export.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace ha
{

class HA_API FeatureExtractor
{
public:
    virtual void compute(cv::InputArray             Image,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray            descriptors) = 0;
};

} // namespace ha