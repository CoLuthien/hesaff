
#include "AffineDeformer.hpp"

#include "Utils.hpp"

#include <opencv2/opencv.hpp>

namespace
{

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
                                      cv::Mat&                      AffineDeformation)
{

    auto const& Img = Pyr[Point.octave_idx].GetLayerBlur(Point.layer_idx);

    float u11 = 1.0f, u12 = 0.0f, u21 = 0.0f, u22 = 1.0f, eig_1 = 1.0f, eig_2 = 1.0f;

    float deform_matrix[4] = {1, 0, 0, 1};

    int const row = Point.row;
    int const col = Point.col;

    float const ratio      = Point.s / (params.initialSigma * Point.pixel_distance);
    int const   maskPixels = params.smmWindowSize * params.smmWindowSize;

    float eigen_ratio_act = 0.0f, eigen_ratio_bef = 0.0f;

    cv::Mat Sample(params.smmWindowSize, params.smmWindowSize, CV_32F);
    cv::Mat grad_x(params.smmWindowSize, params.smmWindowSize, CV_32F);
    cv::Mat grad_y(params.smmWindowSize, params.smmWindowSize, CV_32F);

    for (int Iter = 0; Iter < params.maxIterations; ++Iter)
    {
        utils::SampleDeformAndInterpolate(Img, {row, col}, deform_matrix, Sample);
        float        a = 0, b = 0, c = 0;
        float const* MaskPtr  = GaussisanMask.ptr<float>(0);
        float*       GradxPtr = grad_x.ptr<float>(0);
        float*       GradyPtr = grad_y.ptr<float>(0);

        utils::ComputeGradient(Sample, grad_x, grad_y);

        // estimate SMM
        for (int i = 0; i < maskPixels; ++i)
        {
            const float v   = (*MaskPtr);
            const float gxx = *GradxPtr;
            const float gyy = *GradyPtr;
            const float gxy = gxx * gyy;

            a += gxx * gxx * v;
            b += gxy * v;
            c += gyy * gyy * v;
            GradxPtr++;
            GradyPtr++;
            MaskPtr++;
        }
        a /= maskPixels;
        b /= maskPixels;
        c /= maskPixels;

        MatrixSqrt(a, b, c, eig_1, eig_2);
        eigen_ratio_bef = eigen_ratio_act;
        eigen_ratio_act = 1 - eig_2 / eig_1;

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
            break;
        }
        eig_1 = Eig.at<float>(0);
        eig_2 = Eig.at<float>(1);
        // leave on too high anisotropy
        if ((eig_1 / eig_2 > 6) || (eig_2 / eig_1 > 6)) break;

        if (eigen_ratio_act < params.convergenceThreshold &&
            eigen_ratio_bef < params.convergenceThreshold)
        {
            cv::Mat(2, 2, CV_32F, deform_matrix).copyTo(AffineDeformation);
            return true;
        }
    }

    return false;
}

bool
AffineDeformer::ExtractAndNormalizeAffinePatch(cv::Mat const&        Img,
                                               CandidatePoint const& Point,
                                               cv::Mat&              Patch)
{
    auto const det = cv::determinant(Point.AffineDeformation);
    assert((det - 1) < 1e-2);
    // half patch size in pixels of image
    float mrScale = std::ceil(Point.s * params.mrSize);

    int   patchImageSize    = 2 * int(mrScale) + 1; // odd size
    float imageToPatchScale = (float)patchImageSize / params.patchSize;
}

} // namespace ha