
#include "HessianAffineDetector.hpp"
#include "Utils.hpp"
#include <memory>
#include <execution>

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

HessianAffineDetector::HessianAffineDetector(cv::Ptr<cv::Feature2D>       Backend,
                                             HessianResponsePyramidParams InPyrParam,
                                             HessianDetectorParams        InHessianParam,
                                             AffineDeformerParams         InAffineParam)
    : PyramidParam(std::move(InPyrParam)), HessianParam(std::move(InHessianParam)),
      AffineParam(std::move(InAffineParam)), m_backend(Backend)
{
    m_detector = std::make_unique<HessianDetector>(HessianParam);
    m_deformer = std::make_unique<AffineDeformer>(AffineParam);
}

std::vector<CandidatePoint>
HessianAffineDetector::detectKeypoints(cv::Mat const& Img, cv::Mat const& Mask)
{
    HessianResponsePyramid const Pyr(Img, {});

    auto&& Candidates = m_detector->DetectCandidates(Pyr);

    std::vector<CandidatePoint> Result;
    std::for_each(std::execution::par,
                  Candidates.begin(),
                  Candidates.end(),
                  [MatSize = Img.size, &Pyr, &detector = m_detector, &deformer = m_deformer, &Mask](
                      CandidatePoint& Point) -> void {
                      if (Mask.size == MatSize)
                      {
                          auto MaskValue = at<uint8_t>(Mask, Point.y_pos, Point.x_pos);
                          if (MaskValue == 0)
                          {
                              return;
                          }
                      }
                      bool DeformationFound =
                          deformer->FindAffineDeformation(Pyr, Point, Point.AffineDeformation);
                      if (DeformationFound)
                      {
                          utils::RetifyAffineDeformation(Point.AffineDeformation);
                          bool PatchExtracted =
                              deformer->ExtractAndNormalizeAffinePatch(Pyr, Point, Point.Patch);
                          if (PatchExtracted)
                          {
                              Point.IsValid = true;
                          }
                      }
                  });

    for (auto&& Point : Candidates)
    {
        if (Point.IsValid)
        {
            Result.emplace_back(std::move(Point));
        }
    }
    return Result;
}

void
HessianAffineDetector::detectAndCompute(cv::InputArray image,
                                        cv::InputArray mask,
                                        CV_OUT std::vector<cv::KeyPoint>& keypoints,
                                        cv::OutputArray                   descriptors,
                                        bool                              useProvidedKeypoints)
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

    std::vector<cv::Mat> Descriptors;

    for (auto& Point : ValidCandidates)
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
        Descriptors.emplace_back(std::move(Desc));

        keypoints.emplace_back(
            cv::KeyPoint(Point.x, Point.y, patch_radius, Point.orientation, Point.response));
    }
    if (Descriptors.empty())
    {
        return;
    }
    cv::Mat Result(Descriptors.size(), Descriptors[0].cols, Descriptors[0].depth());
    cv::vconcat(Descriptors, Result);
    Result.copyTo(descriptors);
}

} // namespace ha