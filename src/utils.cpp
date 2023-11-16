
#include "Utils.hpp"

namespace
{

template <typename T>
T
at(cv::Mat const& Img, int const Row, int const Col)
{
    auto const* Ptr = Img.data;
    // return ((T*)(Ptr + Img.step[0] * Row))[Col];
    return Img.at<T>(Row, Col);
}

} // namespace

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

    return std::move(Result);
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
SampleDeformAndInterpolate(cv::Mat const&  Img,
                           cv::Point const Center,
                           float           DeformMatrix[4],
                           cv::Mat&        Result)
{
    int const SampleWidth  = Img.cols - 1;
    int const SampleHeight = Img.rows - 1;
    // output size
    int const OutputWidth  = Result.cols >> 1;
    int const OutputHeight = Result.rows >> 1;

    float const center_row = Center.x;
    float const center_col = Center.y;

    float a11 = DeformMatrix[0], a12 = DeformMatrix[1], a21 = DeformMatrix[2],
          a22 = DeformMatrix[3];

    float* out = Result.ptr<float>(0);
    for (int Row = -OutputHeight; Row <= OutputHeight; ++Row)
    {
        float const rx = center_row + Row * a12;
        float const ry = center_col + Row * a22;
        for (int Col = -OutputWidth; Col <= OutputWidth; ++Col)
        {
            float     row_weight = rx + Col * a11;
            float     col_weight = ry + Col * a21;
            int const x          = (int)std::floor(row_weight);
            int const y          = (int)std::floor(col_weight);

            int clamed_x = std::clamp(x, 0, SampleWidth - 1);
            int clamed_y = std::clamp(y, 0, SampleHeight - 1);

            if (clamed_x == x && clamed_y == y)
            {
                row_weight = row_weight - x;
                col_weight = col_weight - y;

                auto const a_term =
                    (1.0f - col_weight) * ((1.0f - row_weight) * at<float>(Img, y, x) +
                                           row_weight * at<float>(Img, y, x + 1));

                auto const b_term = (col_weight) * ((1.0f - row_weight) * at<float>(Img, y + 1, x) +
                                                    row_weight * at<float>(Img, y + 1, x + 1));

                *out = a_term + b_term;
                out++;
            }
            else
            {
                *out = 0;
                out++;
            }
        }
    }

    return;
}

void
ComputeGradient(cv::Mat const& Img, cv::Mat& Gradx, cv::Mat Grady)
{
    cv::Sobel(Img, Gradx, Img.depth(), 1, 0);
    cv::Sobel(Img, Grady, Img.depth(), 0, 1);
}

cv::Mat
ComputeGaussianMask(std::size_t const size)
{
    std::size_t MaskSize = size;
    if (size % 2 != 1)
    {
        MaskSize += 1;
    }
    cv::Mat Mask(MaskSize, MaskSize, CV_32FC1);

    float sigma  = MaskSize / 3.0f;
    float sigma2 = -2.0f * std::pow(sigma, 2);

    auto const size_x = Mask.cols;
    auto const size_y = Mask.rows;

    auto const half_x = (size_x - 1) / 2;
    auto const half_y = (size_y - 1) / 2;

    for (int row = 0; row < Mask.rows; ++row)
    {
        for (int col = 0; col < Mask.cols; ++col)
        {
            /*
            to achieve
            0 1 2 3 4 ... half_x - 1 , half_x, half_x -1 , .... 3, 2, 1, 0
            same for y
            */
            auto const x             = std::abs(std::abs(half_x - row) - half_x);
            auto const y             = std::abs(std::abs(half_y - col) - half_y);
            auto const Numerator     = -(std::pow(x, 2) + std::pow(y, 2));
            Mask.at<float>(row, col) = std::exp(Numerator / sigma2);
        }
    }

    Mask /= Mask.at<float>(half_x, half_y);

    return Mask;
}

void
RetifyAffineDeformation(float deformation[4])
{
    double a = deformation[0];
    double b = deformation[1];
    double c = deformation[2];
    double d = deformation[3];

    double det  = std::sqrt(std::abs(a * d - b * c));
    double b2a2 = std::sqrt(b * b + a * a);

    deformation[0] = b2a2 / det;
    deformation[1] = 0;
    deformation[2] = (d * b + c * a) / (b2a2 * det);
    deformation[3] = det / b2a2;
}

} // namespace utils