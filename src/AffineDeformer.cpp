
#include "AffineDeformer.hpp"

#include "Utils.hpp"

#include <opencv2/opencv.hpp>
#include <cmath>

namespace
{

template <typename T>
T
at(cv::Mat const& Img, int const width, int const Height)
{
    auto const* Ptr = Img.data;
    return ((T*)(Ptr + Img.step[0] * width))[Height];
}

} // namespace

namespace
{

// 이게 뭔지 해독해서 풀기
void
MatrixSqrt(float& a, float& b, float& c, float& l1, float& l2)
{
    double t, r;
    if (b != 0)
    {
        r = double(c - a) / (2 * b);
        if (r >= 0)
            t = 1.0 / (r + std::sqrt(1 + r * r));
        else
            t = -1.0 / (-r + std::sqrt(1 + r * r));
        r = 1.0 / std::sqrt(1 + t * t); /* c */
        t = t * r;                      /* s */
    }
    else
    {
        r = 1;
        t = 0;
    }
    double x, z, d;

    x = 1.0 / std::sqrt(r * r * a - 2 * r * t * b + t * t * c);
    z = 1.0 / std::sqrt(t * t * a + 2 * r * t * b + r * r * c);

    d = std::sqrt(x * z);
    x /= d;
    z /= d;
    // let l1 be the greater eigenvalue
    if (x < z)
    {
        l1 = float(z);
        l2 = float(x);
    }
    else
    {
        l1 = float(x);
        l2 = float(z);
    }
    // output square root
    a = float(r * r * x + t * t * z);
    b = float(-r * t * x + t * r * z);
    c = float(t * t * x + r * r * z);
}

} // namespace

namespace ha
{
AffineDeformer::AffineDeformer() : GaussisanMask(utils::ComputeGaussianMask(params.smmWindowSize))
{
}

AffineDeformer::AffineDeformer(AffineDeformerParams InParams)
    : AffineDeformer(InParams.maxIterations,
                     InParams.smmWindowSize,
                     InParams.patchSize,
                     InParams.convergenceThreshold,
                     InParams.initialSigma,
                     InParams.mrSize)
{
}
AffineDeformer::AffineDeformer(int const   NIteration,
                               int const   SmmWindowSize,
                               int const   PatchSize,
                               float const ConvergeThreshold,
                               float const StartSigma,
                               float const MRSize)
    : params({.maxIterations        = NIteration,
              .smmWindowSize        = SmmWindowSize,
              .patchSize            = PatchSize,
              .convergenceThreshold = ConvergeThreshold,
              .initialSigma         = StartSigma,
              .mrSize               = MRSize}),
      GaussisanMask(utils::ComputeGaussianMask(params.smmWindowSize))
{
}

bool
AffineDeformer::FindAffineDeformation(HessianResponsePyramid const& Pyr,
                                      CandidatePoint const&         Point,
                                      cv::Mat&                      AffineDeformation) const
{

    auto const& Img = Pyr[Point.octave_idx].GetLayerBlur(Point.layer_idx);

    float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, eig_1 = 1.0f, eig_2 = 1.0f;

    float deform_matrix[4] = {1, 0, 0, 1};

    int const width  = Point.x_pos;
    int const height = Point.y_pos;

    float const ratio = Point.s / (params.initialSigma * Point.pixel_distance);

    float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;

    for (int Iter = 0; Iter < params.maxIterations; ++Iter)
    {
        cv::Mat Sample(params.smmWindowSize, params.smmWindowSize, CV_32F);
        cv::Mat grad_x;
        cv::Mat grad_y;

        float TempDeform[4] = {u11 * ratio, u12 * ratio, u21 * ratio, u22 * ratio};
        utils::SampleDeformAndInterpolate(Img, {width, height}, TempDeform, Sample);
        utils::ComputeGradient(Sample, grad_x, grad_y);

        // estimate SMM, second moment matrix
        auto SMM = utils::EstimateStructureTensor(GaussisanMask, grad_x, grad_y);

        float a = 0, b = 0, c = 0;
        a = at<float>(SMM, 0, 0);
        b = at<float>(SMM, 0, 1);
        c = at<float>(SMM, 1, 1);

        float Eig1, Eig2;
        MatrixSqrt(a, b, c, Eig1, Eig2);

        eigen_ratio_bef = eigen_ratio_act;
        eigen_ratio_act = 1 - Eig2 / Eig1;

        float u11t = u11, u12t = u12;

        u11 = a * u11t + b * u21;
        u12 = a * u12t + b * u22;
        u21 = b * u11t + c * u21;
        u22 = b * u12t + c * u22;

        // clang-format off
        deform_matrix[0] = u11; deform_matrix[1] = u12;
        deform_matrix[2] = u21; deform_matrix[3] = u22;
        // clang-format on
        cv::Mat Eig;
        cv::Mat Deform(2, 2, CV_32F, deform_matrix);
        if (cv::eigen(Deform, Eig) == false)
        {
            return false;
        }
        auto const* EigPtr = Eig.ptr<float>();

        eig_1 = EigPtr[0];
        eig_2 = EigPtr[1];
        // leave on too high anisotropy
        if ((eig_1 / eig_2 > 6.) | (eig_2 / eig_1 > 6.))
        {
            return false;
        }

        if (eigen_ratio_act < params.convergenceThreshold &
            eigen_ratio_bef < params.convergenceThreshold)
        {
            AffineDeformation = (cv::Mat_<float>(2, 2) << u11, u12, u21, u22);
            return true;
        }
    }

    return false;
}

bool
AffineDeformer::ExtractAndNormalizeAffinePatch(HessianResponsePyramid const& Pyr,
                                               CandidatePoint const&         Point,
                                               cv::Mat&                      Patch) const
{

    auto const& Img = Pyr[Point.octave_idx].GetLayerBlur(Point.layer_idx);
    assert((cv::determinant(Point.AffineDeformation) - 1) < 1e-2);
    // half patch size in pixels of image
    float mrScale = std::ceil(Point.s * params.mrSize);

    int   patchImageSize    = 2 * std::ceil(mrScale) + 1; // odd size
    float imageToPatchScale = (float)patchImageSize / params.patchSize;

    int const ImgWidth  = Img.cols;
    int const ImgHeight = Img.rows;

    cv::Point const Center{Point.x_pos, Point.y_pos};
    if (imageToPatchScale > 0.4)
    {
        // the pixels in the image are 0.4 apart + the affine deformation
        // leave +1 border for the bilinear interpolation
        patchImageSize += 2;
        cv::Mat     Sample(patchImageSize, patchImageSize, CV_32F);
        auto const* DeformMatrix = Point.AffineDeformation.ptr<float>();

        auto TouchBorder = utils::SampleDeformAndInterpolate(Img, Center, DeformMatrix, Sample);
        if (TouchBorder == false)
        {
            cv::Mat Result(params.patchSize, params.patchSize, CV_32F);

            // auto&& Blur = utils::GaussianBlurRelativeKernel(Sample, 1.5 * imageToPatchScale);

            auto&& Blur = utils::KuwaharaFilter(Sample, static_cast<int>(3 * imageToPatchScale));
            float const Deform[4] = {imageToPatchScale, 0, 0, imageToPatchScale};

            TouchBorder = utils::SampleDeformAndInterpolate(
                Blur, {patchImageSize / 2, patchImageSize / 2}, Deform, Result);

            Result.copyTo(Patch);
            return TouchBorder == false;
        }
        else
        {
            return false;
        }
    }
    else
    {
        cv::Mat     Tmp          = Point.AffineDeformation * imageToPatchScale;
        auto const* DeformMatrix = Tmp.ptr<float>();

        cv::Mat Result(params.patchSize, params.patchSize, CV_32F);
        utils::SampleDeformAndInterpolate(Img, Center, DeformMatrix, Result);
        Result.copyTo(Patch);
    }
    return true;
}

} // namespace ha