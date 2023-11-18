
#include "HessianAffineDetector.hpp"

#include <memory>
#include <execution>

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
HessianAffineDetector::detectKeypoints(cv::Mat const& Img)
{
    HessianResponsePyramid const Pyr(Img, {});

    auto&& Candidates = m_detector->DetectCandidates(Pyr);

    std::vector<CandidatePoint> Result;
    std::for_each(std::execution::par,
                  Candidates.begin(),
                  Candidates.end(),
                  [&Pyr, &detector = m_detector, &deformer = m_deformer](CandidatePoint& Point) {
                      bool DeformationFound =
                          deformer->FindAffineDeformation(Pyr, Point, Point.AffineDeformation);
                      if (DeformationFound)
                      {
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

    auto&& ValidCandidates = detectKeypoints(image.getMat());

    std::vector<float> Descs;

    for (auto& Point : ValidCandidates)
    {
        auto const x = (float)Point.Patch.cols / 2;
        auto const y = (float)Point.Patch.rows / 2;

        auto const patch_radius = std::sqrt(2 * x * x) / 2;

        cv::Size    PatchSize{Point.Patch.cols, Point.Patch.rows};
        std::vector Location{
            cv::KeyPoint{x, y, patch_radius, -1, Point.response, Point.octave_idx},
        };
        cv::Mat Desc;
        cv::Mat Img;
        cv::normalize(Point.Patch, Img, 255, 0, cv::NORM_MINMAX, CV_8U);
        m_backend->compute(Img, Location, Desc);
        // std::cout << Desc << '\n';
        //  std::cout << Img << '\n';

        keypoints.emplace_back(
            cv::KeyPoint{Point.x, Point.y, patch_radius, -1, Point.response, Point.octave_idx});
        auto const* ptr = Desc.ptr<float>();
        for (int i = 0; i < Desc.cols; i++)
        {
            Descs.push_back(ptr[i]);
        }
    }

    cv::Mat Result(ValidCandidates.size(), 128, CV_32F, Descs.data());

    Result.copyTo(descriptors);
}

} // namespace ha