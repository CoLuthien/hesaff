
#include "Utils.hpp"

namespace
{

template <typename T>
T&
at(cv::Mat const& Img, int const width, int const Height)
{
    auto const* Ptr = Img.data;
    return ((T*)(Ptr + Img.step[0] * width))[Height];
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
    cv::GaussianBlur(Img, Result, {size, size}, Sigma, Sigma, cv::BORDER_ISOLATED);

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

    return std::move((Jxx.mul(Jyy) - Jxy.mul(Jxy)) * norm);
}

bool
IsRegionMax(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t const Col)
{
    bool       result  = true;
    auto const RowIter = Row + 2;
    auto const ColIter = Col + 2;
    for (int r = Row - 1; r != RowIter; ++r)
    {
        const float* row = Img.ptr<float>(r);
        for (int c = Col - 1; c != ColIter; ++c)
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
    bool       result  = true;
    auto const RowIter = Row + 2;
    auto const ColIter = Col + 2;
    for (int r = Row - 1; r != RowIter; ++r)
    {
        const float* row = Img.ptr<float>(r);
        for (int c = Col - 1; c != ColIter; ++c)
        {
            if (row[c] < Value)
            {
                result = false;
            }
        }
    }

    return result;
}

bool
SampleDeformAndInterpolate(cv::Mat const&    Img,
                           cv::Point2i const Center,
                           float const       DeformMatrix[4],
                           cv::Mat&          Result)
{
    int const SampleWidth  = Img.cols - 1;
    int const SampleHeight = Img.rows - 1;
    // output size
    int const RightEnd = (Result.cols >> 1) + 1;
    int const TopEnd   = (Result.rows >> 1) + 1;

    int const LeftEnd   = -(Result.cols >> 1);
    int const BottomEnd = -(Result.rows >> 1);

    float const center_row = (float)Center.x;
    float const center_col = (float)Center.y;

    float const a11 = DeformMatrix[0], a12 = DeformMatrix[1], a21 = DeformMatrix[2],
                a22 = DeformMatrix[3];

    float* out           = Result.ptr<float>(0);
    bool   BoundaryTouch = false;
    for (int Row = BottomEnd; Row < TopEnd; ++Row)
    {
        float const rx = center_row + Row * a12;
        float const ry = center_col + Row * a22;
        for (int Col = LeftEnd; Col < RightEnd; ++Col)
        {
            float     row_weight = rx + Col * a11;
            float     col_weight = ry + Col * a21;
            int const x          = (int)std::floor(row_weight);
            int const y          = (int)std::floor(col_weight);

            int clamed_x = std::clamp(x, 0, SampleWidth - 1);
            int clamed_y = std::clamp(y, 0, SampleHeight - 1);

            if (clamed_x == x && clamed_y == y)
            {
                row_weight        = row_weight - x;
                col_weight        = col_weight - y;
                auto const a_coef = (1.0f - col_weight);

                auto const a_term = a_coef * ((1.0f - row_weight) * at<float>(Img, y, x) +
                                              row_weight * at<float>(Img, y, x + 1));

                auto const b_term = (col_weight) * ((1. - row_weight) * at<float>(Img, y + 1, x) +
                                                    row_weight * at<float>(Img, y + 1, x + 1));

                *out = a_term + b_term;
                out++;
            }
            else
            {
                *out = 0;
                out++;
                BoundaryTouch = true;
            }
        }
    }

    return BoundaryTouch;
}

void
ComputeGradient(cv::Mat const& Img, cv::Mat& Gradx, cv::Mat& Grady)
{
    cv::Scharr(Img, Gradx, Img.depth(), 1, 0);
    cv::Scharr(Img, Grady, Img.depth(), 0, 1);
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

    float sigma  = MaskSize / 6.0f;
    float sigma2 = -2.0f * std::pow(sigma, 2);

    auto const size_x = Mask.cols;
    auto const size_y = Mask.rows;

    auto const half_x = (size_x - 1) / 2;
    auto const half_y = (size_y - 1) / 2;

    for (int row = 0; row < size_y; ++row)
    {
        for (int col = 0; col < size_x; ++col)
        {
            /*
            to achieve
            0 1 2 3 4 ... half_x - 1 , half_x, half_x -1 , .... 3, 2, 1, 0
            same for y
            */
            auto const x = std::abs(std::abs(half_x - row) - half_x);
            auto const y = std::abs(std::abs(half_y - col) - half_y);

            auto const Numerator      = -(std::pow(x, 2) + std::pow(y, 2));
            at<float>(Mask, row, col) = std::exp(Numerator / sigma2);
        }
    }

    Mask /= at<float>(Mask, half_x, half_y);

    return Mask;
}

void
RetifyAffineDeformation(cv::Mat& Deformation)
{
    auto* deformation = Deformation.ptr<float>();

    double a = deformation[0];
    double b = deformation[1];
    double c = deformation[2];
    double d = deformation[3];

    double det  = cv::determinant(Deformation);
    double b2a2 = std::sqrt(b * b + a * a);

    deformation[0] = b2a2 / det;
    deformation[1] = 0;
    deformation[2] = (d * b + c * a) / (b2a2 * det);
    deformation[3] = det / b2a2;
}

bool
IsSampleTouchBorder(cv::Size const  ImgSize,
                    cv::Size const  SampleSize,
                    cv::Point const Center,
                    float const     DeformMatrix[4])
{
    const float width      = static_cast<float>(ImgSize.width);
    const float height     = static_cast<float>(ImgSize.height);
    const float halfWidth  = static_cast<float>(SampleSize.width >> 1);
    const float halfHeight = static_cast<float>(SampleSize.height >> 1);

    float x[4] = {-halfWidth, -halfWidth, +halfWidth, +halfWidth};
    float y[4] = {-halfHeight, +halfHeight, -halfHeight, +halfHeight};

    float a11 = DeformMatrix[0], a12 = DeformMatrix[1], a21 = DeformMatrix[2],
          a22 = DeformMatrix[3];

    float const ofsx = Center.x;
    float const ofsy = Center.y;
    for (int i = 0; i < 4; i++)
    {
        float imx = ofsx + x[i] * a11 + y[i] * a12;
        float imy = ofsy + x[i] * a21 + y[i] * a22;
        if (floor(imx) <= 0 || floor(imy) <= 0 || ceil(imx) >= width || ceil(imy) >= height)
            return true;
    }
    return false;
}

cv::Mat
EstimateStructureTensor(cv::Mat const& Window, cv::Mat const& GradX, cv::Mat const& GradY)
{
    auto area = Window.cols * Window.rows;
    auto a    = cv::sum(GradX.mul(GradX).mul(Window)) / area;
    auto b    = cv::sum(GradY.mul(GradY).mul(Window)) / area;
    auto c    = cv::sum(GradX.mul(GradY).mul(Window)) / area;

    return (cv::Mat_<double>(2, 2) << a[0], c[0], c[0], b[0]);
}

void
PhotometricallyNormalizeImage(cv::InputArray Patch, cv::OutputArray Normalized)
{
    auto Img = Patch.getMat();

    cv::Mat tmp;
    auto    mean = cv::mean(Img);
    double  min, max;
    cv::minMaxLoc(Img, &min, &max);

    auto fac = 1. / (max - min);
    cv::exp(-(Img - mean) * fac, tmp);

    auto Denorm = 1 / (1 + tmp);

    cv::Mat Out = 255 * Denorm;
    Out.copyTo(Normalized);
}

// matrix square root by eigen decomposition
void
MatrixSqrt(const cv::Mat& matrix, cv::Mat& sqrtMatrix, cv::Mat& EigenValues)
{
    cv::Mat eigenvectors, sqrtvalues, result;
    cv::eigen(matrix, EigenValues, eigenvectors);

    // Calculate square root of the eigenvalues
    cv::sqrt(EigenValues, sqrtvalues);

    // Reconstruct the square root matrix
    result = (eigenvectors * cv::Mat::diag(sqrtvalues) * eigenvectors.inv());

    cv::normalize(result, sqrtMatrix);
}

} // namespace utils