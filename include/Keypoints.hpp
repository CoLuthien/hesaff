
#pragma once

#include "ha_export.hpp"

#include <opencv2/opencv.hpp>

namespace ha
{
struct HA_API CandidatePoint
{
public:
    float x, y, s;
    float response;
    float pixel_distance;
    int   type;

public:
    cv::Mat AffineDeformation;
    cv::Mat Patch;

public:
    int row, col;

public:
    int octave_idx, layer_idx;
};

} // namespace ha
