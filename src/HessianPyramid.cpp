
#include "Utils.hpp"
#include "HessianPyramid.hpp"

namespace ha
{
HessianResponseOctave::HessianResponseOctave(cv::Mat const& Image,
                                             int const      nLayers,
                                             float const    InPixelDist,
                                             float const    InitialSigma,
                                             float const    SigmaStep)
    : numLayers(nLayers), PixelDistance(InPixelDist)
{
    auto       CurSigma = InitialSigma;
    auto const Step     = SigmaStep;

    auto&&  Current  = utils::HessianResponse(Image, CurSigma);
    cv::Mat NextBlur = utils::GaussianBlurRelativeKernel(Image, CurSigma);

    m_layers.emplace_back(std::move(Current));

    for (int Layer = 1; Layer < nLayers; ++Layer)
    {
        // compute the increase necessary for the next level and compute the next level
        NextBlur = utils::GaussianBlurRelativeKernel(NextBlur, CurSigma);
        Current  = utils::HessianResponse(NextBlur, CurSigma);
        m_sigmas.emplace_back(CurSigma);
        m_layers.emplace_back(std::move(Current));
        m_blurs.emplace_back(NextBlur);
        CurSigma *= Step;
    }
}
HessianResponsePyramid ::HessianResponsePyramid(cv::Mat const& Image,
                                                int const      nOctaves,
                                                int const      nLayers,
                                                float const    inSigma)
    : params({.numOctaves = nOctaves, .numLayers = nLayers, .initialSigma = inSigma})
{
    auto const InitialSigma         = 0.5;
    auto const InitialPixelDistance = 1.;

    cv::Mat Img;

    if (params.initialSigma > InitialSigma)
    {
        auto const sigma = std::sqrt(std::pow(params.initialSigma, 2) - std::pow(InitialSigma, 2));

        Img = utils::GaussianBlurRelativeKernel(Image, sigma);
    }
    else
    {
        Img = Image.clone();
    }
    auto const Step = std::pow(2., 1. / params.numLayers);

    float pix_dist = InitialPixelDistance;
    for (auto Octave = 0; Octave < params.numOctaves; ++Octave)
    {
        m_octaves.emplace_back(
            HessianResponseOctave(Img, params.numLayers, pix_dist, params.initialSigma, Step));

        { // half size the image
            cv::Mat Resized;
            cv::resize(Img, Resized, {0, 0}, 0.5, 0.5, cv::INTER_AREA);
            Img = std::move(Resized);
            pix_dist *= 2;
        }
    }
}

} // namespace ha