
#pragma once
#include "ha_export.hpp"

#include <opencv2/opencv.hpp>

namespace std
{
template <>
struct hash<cv::Point>
{
    std::size_t operator()(cv::Point const& p) const noexcept
    {
        std::size_t h1 = std::hash<float>{}(p.x);
        std::size_t h2 = std::hash<float>{}(p.y);
        return (h1 ^ h2);
    }
};

} // namespace std

namespace utils
{
HA_API
cv::Mat GaussianBlurRelativeKernel(cv::Mat const& Img, float const Sigma);

HA_API
cv::Mat HessianResponse(cv::Mat const& Img, float const Sigma);

HA_API
bool IsRegionMax(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t Col);

HA_API
bool IsRegionMin(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t Col);

HA_API
void solveLinear3x3(float* A, float* b);

} // namespace utils