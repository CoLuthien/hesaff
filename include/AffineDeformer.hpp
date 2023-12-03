
#pragma once

#include "ha_export.hpp"
#include "Keypoints.hpp"
#include "HessianPyramid.hpp"

#include <opencv2/opencv.hpp>

namespace ha
{
struct HA_API AffineDeformerParams
{
    // number of affine shape interations
    int const maxIterations = 16;

    // convergence threshold, i.e. maximum deviation from isotropic shape at convergence

    // widht and height of the SMM mask
    int const smmWindowSize = 19;

    // width and height of the patch
    int const patchSize = 41;

    // amount of smoothing applied to the initial level of first octave
    float const convergenceThreshold = .05;
    float const initialSigma         = 3.;
    // size of the measurement region (as multiple of the feature scale)
    float const mrSize = 3.f * std::sqrtf(3);
};

class HA_API AffineDeformer
{
public:
    AffineDeformer();
    AffineDeformer(AffineDeformerParams InParams);
    AffineDeformer(int const   NIteration,
                   int const   SmmWindowSize,
                   int const   PatchSize,
                   float const ConvergeThreshold,
                   float const StartSigma,
                   float const MRSize);

public:
    bool FindAffineDeformation(HessianResponsePyramid const& Pyr,
                               CandidatePoint const&         Point,
                               cv::Mat&                      AffineDeformation) const;

    bool ExtractAndNormalizeAffinePatch(HessianResponsePyramid const& Pyr,
                                        CandidatePoint const&         Point,
                                        cv::Mat&                      Patch) const;

private:
    AffineDeformerParams const params;
    cv::Mat const              GaussisanMask;
};

} // namespace ha