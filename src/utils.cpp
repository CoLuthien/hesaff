
#include "Utils.hpp"

namespace
{

template <typename T>
inline T&
at(cv::Mat const& Img, int const Row, int const Col)
{
    auto const* Ptr = Img.data;
    return ((T*)(Ptr + Img.step[0] * Row))[Col];
}

} // namespace

namespace utils
{

cv::Mat
KuwaharaFilter(cv::Mat const& Img, int m)
{
    int quadrants_size = std::clamp(m / 2, 3, m);

    static constexpr auto c1 = [](int x, int y, int quad_size, int x_max, int y_max) {
        auto const offset = (quad_size / 2) + 1;

        int px = std::clamp(x - offset, 0, x_max - quad_size);
        int py = std::clamp(y - offset, 0, y_max - quad_size);
        return cv::Point{px, py};
    };
    static constexpr auto c2 = [](int x, int y, int quad_size, int x_max, int y_max) {
        auto const offset = (quad_size / 2) + 1;

        int px = std::clamp(x, 0, x_max - quad_size);
        int py = std::clamp(y - offset, 0, y_max - quad_size);
        return cv::Point{px, py};
    };
    static constexpr auto c3 = [](int x, int y, int quad_size, int x_max, int y_max) {
        auto const offset = (quad_size / 2) + 1;

        int px = std::clamp(x - offset, 0, x_max - quad_size);
        int py = std::clamp(y, 0, y_max - quad_size);
        return cv::Point{px, py};
    };
    static constexpr auto c4 = [](int x, int y, int quad_size, int x_max, int y_max) {
        int px = std::clamp(x, 0, x_max - quad_size);
        int py = std::clamp(y, 0, y_max - quad_size);
        return cv::Point{px, py};
    };

    std::unordered_map<cv::Rect, std::pair<double, double>> calculated;

    cv::Size quad_size{quadrants_size, quadrants_size};

    static constexpr auto mean_and_std =
        [](cv::Mat const& Img, cv::Rect ROI, decltype(calculated)& map) {
            if (map.contains(ROI))
            {
                return map[ROI];
            }
            else
            {
                auto&&  Mat  = Img(ROI);
                auto    mean = cv::mean(Mat);
                cv::Mat deviation;
                cv::pow(Mat - mean, 2, deviation);
                auto factor = 1. / Mat.total();
                deviation *= factor;

                auto stddev = std::sqrt(cv::sum(deviation)[0]);

                map.emplace(ROI, std::make_pair(mean[0], stddev));

                return std::make_pair(mean[0], stddev);
            }
        };

    auto const rows        = Img.rows;
    auto const cols        = Img.cols;
    auto const target_rows = Img.rows - quadrants_size;
    auto const target_cols = Img.cols - quadrants_size;

    cv::Mat Result(Img.size(), CV_32F, 0.);
    calculated.reserve(Img.total());

    for (int y = 0; y < rows; ++y)
    {
        auto* values = Result.ptr<float>(y);
        for (int x = 0; x < cols; ++x)
        {
            cv::Rect r1{c1(x, y, quadrants_size, cols, rows), quad_size};

            auto [m1, s1] = mean_and_std(Img, r1, calculated);

            cv::Rect r2{c2(x, y, quadrants_size, cols, rows), quad_size};

            auto [m2, s2] = mean_and_std(Img, r2, calculated);

            cv::Rect r3{c3(x, y, quadrants_size, cols, rows), quad_size};

            auto [m3, s3] = mean_and_std(Img, r3, calculated);

            cv::Rect r4{c4(x, y, quadrants_size, cols, rows), quad_size};

            auto [m4, s4] = mean_and_std(Img, r4, calculated);

            std::array sigmas{s1, s2, s3, s4};
            std::array means{m1, m2, m3, m4};

            auto Idx =
                std::distance(sigmas.begin(), std::min_element(sigmas.begin(), sigmas.end()));

            values[x] = means[Idx];
        }
    }
    return Result.clone();
}

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

    auto const norm = std::pow(Sigma, 4.);

    return norm * (Jxx.mul(Jyy) - Jxy.mul(Jxy));
}

bool
IsRegionMax(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t const Col)
{
    return (at<float>(Img, Row, Col) > Value) == false;
}

bool
IsRegionMin(cv::Mat const& Img, float const Value, std::size_t const Row, std::size_t Col)
{
    return (at<float>(Img, Row, Col) < Value) == false;
}

bool
SampleDeformAndInterpolate(cv::Mat const&    Img,
                           cv::Point2i const Center,
                           float const       DeformMatrix[4],
                           cv::Mat&          Result)
{
    int const SampleWidth  = Img.cols - 1;
    int const SampleHeight = Img.rows - 1;
    int const RightEnd     = (Result.cols >> 1);
    int const TopEnd       = (Result.rows >> 1);
    int const LeftEnd      = -(Result.cols >> 1);
    int const BottomEnd    = -(Result.rows >> 1);

    float const center_row = static_cast<float>(Center.x);
    float const center_col = static_cast<float>(Center.y);

    float const a11 = DeformMatrix[0], a12 = DeformMatrix[1], a21 = DeformMatrix[2],
                a22 = DeformMatrix[3];

    cv::Mat map_x(Result.size(), CV_32F);
    cv::Mat map_y(Result.size(), CV_32F);

    bool TouchBorder = false;

    for (int Row = BottomEnd; Row < TopEnd + 1; ++Row)
    {
        for (int Col = LeftEnd; Col < RightEnd + 1; ++Col)
        {
            float rx = center_row + Row * a12 + Col * a11;
            float ry = center_col + Row * a22 + Col * a21;

            auto const x = std::floor(rx);
            auto const y = std::floor(ry);

            if ((x < 0 or y < 0) or (x > SampleWidth or y > SampleHeight))
            {
                TouchBorder = true;
            }

            at<float>(map_x, Row + TopEnd, Col + RightEnd) = rx;
            at<float>(map_y, Row + TopEnd, Col + RightEnd) = ry;
        }
    }
    cv::remap(Img, Result, map_x, map_y, cv::INTER_AREA, cv::BORDER_CONSTANT);

    return TouchBorder;
}

void
ComputeGradient(cv::Mat const& Img, cv::Mat& Gradx, cv::Mat& Grady)
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

    float sigma      = MaskSize / 6.0f;
    float sigma2_inv = 1. / (2.0f * std::pow(sigma, 2));

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
            auto const x = std::abs(std::abs(half_x - row));
            auto const y = std::abs(std::abs(half_y - col));

            auto const Numerator      = (std::pow(x, 2) + std::pow(y, 2));
            at<float>(Mask, row, col) = std::exp(-Numerator * sigma2_inv);
        }
    }

    auto const factor = 1. / at<float>(Mask, half_x, half_y);

    Mask *= factor;

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
    auto area = 1. / Window.cols * Window.rows;

    cv::Scalar_<float> a = cv::sum(GradX.mul(GradX).mul(Window)) * area;
    cv::Scalar_<float> b = cv::sum(GradY.mul(GradY).mul(Window)) * area;
    cv::Scalar_<float> c = cv::sum(GradX.mul(GradY).mul(Window)) * area;

    return (cv::Mat_<float>(2, 2) << a[0], c[0], c[0], b[0]);
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