
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
    HessianAffineDetector(cv::Ptr<cv::Feature2D>       Backend,
                          HessianResponsePyramidParams InPyrParam     = {},
                          HessianDetectorParams        InHessianParam = {},
                          AffineDeformerParams         InAffineParam  = {});

public:
    void detectAndCompute(cv::InputArray image,
                          cv::InputArray mask,
                          CV_OUT std::vector<cv::KeyPoint>& keypoints,
                          cv::OutputArray                   descriptors,
                          bool                              useProvidedKeypoints = false);

private:
    std::vector<CandidatePoint> detectKeypoints(cv::Mat const& Img, cv::Mat const& Mask);

private:
    HessianResponsePyramidParams PyramidParam;
    HessianDetectorParams        HessianParam;
    AffineDeformerParams         AffineParam;

private:
    cv::Ptr<cv::Feature2D>           m_backend;
    std::shared_ptr<HessianDetector> m_detector;
    std::shared_ptr<AffineDeformer>  m_deformer;
};

} // namespace ha
