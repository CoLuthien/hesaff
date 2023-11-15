
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

void
solveLinear3x3(float* A, float* b)
{
    // find pivot of first column
    int    i   = 0;
    float* pr  = A;
    float  vp  = abs(A[0]);
    float  tmp = abs(A[3]);
    if (tmp > vp)
    {
        // pivot is in 1st row
        pr = A + 3;
        i  = 1;
        vp = tmp;
    }
    if (abs(A[6]) > vp)
    {
        // pivot is in 2nd row
        pr = A + 6;
        i  = 2;
    }

    // swap pivot row with first row
    if (pr != A)
    {
        std::swap(pr[0], A[0]);
        std::swap(pr[1], A[1]);
        std::swap(pr[2], A[2]);
        std::swap(b[i], b[0]);
    }

    // fixup elements 3,4,5,b[1]
    vp = A[3] / A[0];
    A[4] -= vp * A[1];
    A[5] -= vp * A[2];
    b[1] -= vp * b[0];

    // fixup elements 6,7,8,b[2]]
    vp = A[6] / A[0];
    A[7] -= vp * A[1];
    A[8] -= vp * A[2];
    b[2] -= vp * b[0];

    // find pivot in second column
    if (std::abs(A[4]) < std::abs(A[7]))
    {
        std::swap(A[7], A[4]);
        std::swap(A[8], A[5]);
        std::swap(b[2], b[1]);
    }

    // fixup elements 7,8,b[2]
    vp = A[7] / A[4];
    A[8] -= vp * A[5];
    b[2] -= vp * b[1];

    // solve b by back-substitution
    b[2] = (b[2]) / A[8];
    b[1] = (b[1] - A[5] * b[2]) / A[4];
    b[0] = (b[0] - A[2] * b[2] - A[1] * b[1]) / A[0];
}
} // namespace utils