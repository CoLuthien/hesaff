
#pragma once

#include "ha_export.hpp"
#include "Keypoints.hpp"
#include "HessianPyramid.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_set>

namespace ha
{
struct HA_API HessianDetectorParams
{
public:
    HessianDetectorParams(int const   inBorderSize          = 9,
                          float const inThreshold           = 5.3,
                          float const inEdgeEigenValueRatio = 10);

public: // input from user
    float const Threshold;
    float const EdgeEigenValueRatio;
    int const   borderSize;

public: // calculated from user input
    float const edgeScoreThreshold;
    float const finalThreshold;
    float const positiveThreshold;
    float const negativeThreshold;
};

class HA_API HessianDetector
{
public:
    enum
    {
        HESSIAN_DARK   = 0,
        HESSIAN_BRIGHT = 1,
        HESSIAN_SADDLE = 2,
    };

public:
    HessianDetector(HessianDetectorParams Params);
    HessianDetector(int const nBorders, float const edgeEigvalRatio, float const threshold);

public:
    std::vector<CandidatePoint> DetectCandidates(HessianResponsePyramid const& Pyr) const;

private:
    std::vector<CandidatePoint> FindOctaveCandidates(HessianResponsePyramid const&  Pyr,
                                                     std::size_t const              Octave,
                                                     std::unordered_set<cv::Point>& VisitMap) const;
    std::vector<CandidatePoint> FindLayerCandidates(HessianResponsePyramid const&  Pyr,
                                                    std::size_t const              Octave,
                                                    std::size_t const              LayerIdx,
                                                    std::unordered_set<cv::Point>& VisitMap) const;

    bool LocalizeCandidate(CandidatePoint&                Point,
                           std::unordered_set<cv::Point>& VisitMap,
                           cv::Mat const&                 Prev,
                           cv::Mat const&                 Current,
                           cv::Mat const&                 Next,
                           int const                      PositionY,
                           int const                      PositionX,
                           float const                    CurSigma,
                           float const                    pixelDistance,
                           int const                      numLayers) const;

private:
    HessianDetectorParams const param_detect;
};

} // namespace ha
