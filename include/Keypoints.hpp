
#pragma once

#include "ha_export.hpp"

#include <opencv2/opencv.hpp>

namespace ha
{
struct HA_API CandidatePoint
{
public:
    bool IsValid = false;

public:
    float x, y, s;
    float response;
    float pixel_distance;
    float orientation;
    int    type;

public:
    cv::Mat AffineDeformation;
    cv::Mat Patch;

public:
    int y_pos, x_pos;

public:
    int octave_idx, layer_idx;
};

} // namespace ha
