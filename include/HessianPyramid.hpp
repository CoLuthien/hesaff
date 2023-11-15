
#pragma once

#include "ha_export.hpp"
#include <opencv2/opencv.hpp>

namespace ha
{
struct HessianResponsePyramidParams
{
    int   numOctaves   = 3;   // number of half scaled image for octaves
    int   numLayers    = 5;   // amount of gaussian blurred image for each Hessian Octave
    float initialSigma = 1.6; // amount of smoothing applied to the initial level of first octave
};

class HA_API HessianResponseOctave
{
public:
    HessianResponseOctave(cv::Mat const& Image,
                          int const      nLayers,
                          float const    inPixelDist,
                          float const    InitialSigma,
                          float const    SigmaStep);

public:
    cv::Mat const& operator[](std::size_t Idx) const { return m_layers[Idx]; }
    float const    GetLayerSigma(std::size_t Idx) const { return m_sigmas[Idx]; }

public:
    int const   numLayers;
    float const PixelDistance;

public:
    std::vector<float>   m_sigmas;
    std::vector<cv::Mat> m_blurs;
    std::vector<cv::Mat> m_layers;
};

class HA_API HessianResponsePyramid
{
public:
    HessianResponsePyramid(cv::Mat const& Image,
                           int const      nOctaves,
                           int const      nLayers,
                           float const    inSigma);

public:
    HessianResponseOctave const& operator[](std::size_t Idx) const { return m_octaves[Idx]; }

private:
    std::vector<HessianResponseOctave> m_octaves;

private:
    HessianResponsePyramidParams const params;
};

} // namespace ha