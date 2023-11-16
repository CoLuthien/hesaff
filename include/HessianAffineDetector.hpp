
#pragma once

#include "ha_export.hpp"

#include "Keypoints.hpp"
#include "AffineDeformer.hpp"
#include "HessianDetector.hpp"

#include <opencv2/opencv.hpp>

namespace ha
{

class HA_API HessianAffineDetector
{
public:
    HessianAffineDetector();

private:
    HessianDetector m_detector;
    AffineDeformer  m_deformer;
};

} // namespace ha
