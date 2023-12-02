
#include "HessianAffineDetector.hpp"
#include "Utils.hpp"
#include <memory>
#include <execution>
#include <chrono>

namespace
{
template <typename T>
T
at(cv::Mat const& Img, int const Row, int const Col)
{
    auto const* Ptr = Img.data;
    return ((T*)(Ptr + Img.step[0] * Row))[Col];
}

} // namespace

namespace ha
{

HessianAffineDetector::HessianAffineDetector(std::shared_ptr<FeatureExtractor> Backend,
                                             HessianResponsePyramidParams      InPyrParam,
                                             HessianDetectorParams             InHessianParam,
                                             AffineDeformerParams              InAffineParam)
    : PyramidParam(std::move(InPyrParam)), HessianParam(std::move(InHessianParam)),
      AffineParam(std::move(InAffineParam)), m_backend(std::move(Backend))
{
    m_detector = std::make_shared<HessianDetector>(HessianParam);
    m_deformer = std::make_shared<AffineDeformer>(AffineParam);
}

std::vector<CandidatePoint>
HessianAffineDetector::detectKeypoints(cv::Mat const& Img, cv::Mat const& Mask) const
{
    HessianResponsePyramid const Pyr(Img, {});

    //auto   from       = std::chrono::steady_clock::now();
    auto&& Candidates = m_detector->DetectCandidates(Pyr);
    //auto   to         = std::chrono::steady_clock::now();

    std::vector<CandidatePoint> Result;
    Result.reserve(Candidates.size());
    for (auto& Point : Candidates)
    {
        bool DeformationFound =
            m_deformer->FindAffineDeformation(Pyr, Point, Point.AffineDeformation);
        if (DeformationFound)
        {
            utils::RetifyAffineDeformation(Point.AffineDeformation);
            bool PatchExtracted =
                m_deformer->ExtractAndNormalizeAffinePatch(Pyr, Point, Point.Patch);
            if (PatchExtracted)
            {
                Result.emplace_back(std::move(Point));
            }
        }
    }
    //std::chrono::duration<double> time = to - from;
    //std::cout << time.count() << '\n';

    return Result;
}

void
HessianAffineDetector::CalculateDescriptors(std::vector<CandidatePoint> const& candidates,
                                            std::vector<cv::KeyPoint>&         keypoints,
                                            cv::OutputArray                    descriptors) const
{
    std::vector<cv::Mat> Descs;
    for (auto& Point : candidates)
    {
        auto const x = (float)Point.Patch.cols / 2;
        auto const y = (float)Point.Patch.rows / 2;

        auto const patch_radius = std::sqrt(2 * x * x) / 2;

        std::vector Location{
            cv::KeyPoint{x, y, patch_radius, Point.orientation, Point.response},
        };
        cv::Mat Desc;
        cv::Mat Img;
        cv::normalize(Point.Patch, Img, 0, 255, cv::NORM_MINMAX, CV_8U);
        m_backend->compute(Img, Location, Desc);
        Descs.emplace_back(std::move(Desc));

        keypoints.emplace_back(
            cv::KeyPoint(Point.x, Point.y, patch_radius, Point.orientation, Point.response));
    }

    cv::vconcat(Descs, descriptors);
}

void
HessianAffineDetector::detectAndCompute(cv::InputArray image,
                                        cv::InputArray mask,
                                        CV_OUT std::vector<cv::KeyPoint>& keypoints,
                                        cv::OutputArray                   descriptors,
                                        bool useProvidedKeypoints) const
{
    auto const& Image = image.getMat();
    auto const& Mask  = mask.getMat();

    cv::Mat Target;
    if (Image.depth() != CV_32FC1)
    {
        Image.convertTo(Target, CV_32FC1);
    }
    else
    {
        Image.copyTo(Target);
    }
    auto&& ValidCandidates = detectKeypoints(Target, Mask);

    CalculateDescriptors(ValidCandidates, keypoints, descriptors);
}

} // namespace ha