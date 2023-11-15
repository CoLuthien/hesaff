
#pragma once

#include "ha_export.hpp"
#include "HessianPyramid.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_set>

namespace ha
{
struct HA_API HessianDetectorParams
{
public:
    HessianDetectorParams(int const   inBorderSize,
                          float const inThreshold,
                          float const inEdgeEigenValueRatio);

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

struct HA_API CandidatePoint
{
public:
    float x, y, s;
    float response;
    float pixel_distance;
    int   type;

public:
    int octave_idx, layer_idx;
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
    HessianDetector(int const   nOctaves,
                    int const   nLayers,
                    int const   nBorders,
                    float const startSigma,
                    float const edgeEigvalRatio,
                    float const threshold);

public:
    std::vector<CandidatePoint> DetectCandidates(cv::Mat const& Image);

private:
    std::vector<CandidatePoint> FindOctaveCandidates(HessianResponseOctave const& Octave);
    std::vector<CandidatePoint> FindLayerCandidates(HessianResponseOctave const&   Octave,
                                                    int const                      LayerIdx,
                                                    std::unordered_set<cv::Point>& VisitMap);

    bool LocalizeCandidate(CandidatePoint&                Point,
                           std::unordered_set<cv::Point>& VisitMap,
                           cv::Mat const&                 Prev,
                           cv::Mat const&                 Current,
                           cv::Mat const&                 Next,
                           int const                      Row,
                           int const                      Col,
                           float const                    CurSigma,
                           float const                    pixelDistance);

private:
    HessianResponsePyramidParams const param_pyr;
    HessianDetectorParams const        param_detect;
};

} // namespace ha
