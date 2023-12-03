
#pragma once
#include "ha_export.hpp"

#include <opencv2/opencv.hpp>

namespace
{
template <typename T, typename... Rest>
void
hash_combine(std::size_t& seed, const T& v, const Rest&... rest)
{
    seed ^= std::hash<T>{}(v) ^ 0x9e3779b9;
    (hash_combine(seed, rest), ...);
}

} // namespace

namespace std
{

template <>
struct hash<cv::Point>
{
    std::size_t operator()(cv::Point const& p) const noexcept
    {
        std::size_t result = 0;
        hash_combine(result, (int)p.x, (int)p.y);
        return result;
    }
};

} // namespace std

namespace utils
{
HA_API
cv::Mat GaussianBlurRelativeKernel(cv::Mat const& Img, float const Sigma);

HA_API cv::Mat HessianResponse(cv::Mat const& Img, float const Sigma);

HA_API
bool IsRegionMax(cv::Mat const&    Img,
                 float const       Value,
                 std::size_t const Row,
                 std::size_t const Col);

HA_API
bool IsRegionMin(cv::Mat const&    Img,
                 float const       Value,
                 std::size_t const Row,
                 std::size_t const Col);

HA_API
bool SampleDeformAndInterpolate(cv::Mat const&  Img,
                                cv::Point const Center,
                                float const     DeformMatrix[4],
                                cv::Mat&        Result);
HA_API
bool IsSampleTouchBorder(cv::Size const  ImgSize,
                         cv::Size const  SampleSize,
                         cv::Point const Center,
                         float const     DeformMatrix[4]);
HA_API
void ComputeGradient(cv::Mat const& Img, cv::Mat& Gradx, cv::Mat& Grady);

HA_API
cv::Mat ComputeGaussianMask(std::size_t const size);

HA_API
void RetifyAffineDeformation(cv::Mat& Deformation);

HA_API
cv::Mat EstimateStructureTensor(cv::Mat const& Window, cv::Mat const& GradX, cv::Mat const& GradY);

HA_API
void PhotometricallyNormalizeImage(cv::InputArray Patch, cv::OutputArray Normalized);

HA_API
void MatrixSqrt(const cv::Mat& matrix, cv::Mat& sqrtMatrix, cv::Mat& EigenValues);

} // namespace utils