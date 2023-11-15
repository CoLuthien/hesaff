
#include "Utils.hpp"

namespace utils
{
cv::Mat
GaussianBlurRelativeKernel(cv::Mat const& Img, float const Sigma)
{
    int size = (int)(2.0 * 3.0 * Sigma + 1.0);
    if (size % 2 == 0)
    {
        size += 1;
    }

    cv::Mat Result;
    cv::GaussianBlur(Img, Result, {size, size}, Sigma);

    return Result;
}

cv::Mat
HessianResponse(cv::Mat const& Img, float const Sigma)
{
    cv::Mat Jxx, Jyy, Jxy;

    cv::Sobel(Img, Jxx, Img.depth(), 2, 0);
    cv::Sobel(Img, Jyy, Img.depth(), 0, 2);
    cv::Sobel(Img, Jxy, Img.depth(), 1, 1);

    auto const norm = std::pow(Sigma, 2);

    return (Jxx.mul(Jyy) - Jxy.mul(Jxy)) * norm;
}

bool
IsRegionMax(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t Col)
{
    bool result = true;
    for (int r = Row - 1; r <= Row + 1; ++r)
    {
        const float* row = Img.ptr<float>(r);
        for (int c = Col - 1; c <= Col + 1; ++c)
        {
            if (row[c] > Value)
            {
                result = false;
            }
        }
    }

    return result;
}

bool
IsRegionMin(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t Col)
{
    bool result = true;
    for (int r = Row - 1; r <= Row + 1; ++r)
    {
        const float* row = Img.ptr<float>(r);
        for (int c = Col - 1; c <= Col + 1; ++c)
        {
            if (row[c] < Value)
            {
                result = false;
            }
        }
    }

    return result;
}

} // namespace utils