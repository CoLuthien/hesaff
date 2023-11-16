
#include "HessianPyramid.hpp"
#include "HessianDetector.hpp"
#include "Utils.hpp"

namespace
{
template <typename T>
T
at(cv::Mat const& Img, int const Row, int const Col)
{
    auto const* Ptr = Img.data;
    return ((T*)(Ptr + Img.step[0] * Row))[Col];
}

int
GetHessianPointType(float const* ptr, float value)
{
    if (value < 0)
    {
        return ha::HessianDetector::HESSIAN_SADDLE;
    }
    else
    {
        // at this point we know that 2x2 determinant is positive
        // so only check the remaining 1x1 subdeterminant
        float Lxx = (ptr[-1] - 2 * ptr[0] + ptr[1]);
        if (Lxx < 0)
        {
            return ha::HessianDetector::HESSIAN_DARK;
        }
        else
        {
            return ha::HessianDetector::HESSIAN_BRIGHT;
        }
    }
}
} // namespace
namespace ha
{

int
getHessianPointType(float* ptr, float value)
{
    if (value < 0)
        return HessianDetector::HESSIAN_SADDLE;
    else
    {
        // at this point we know that 2x2 determinant is positive
        // so only check the remaining 1x1 subdeterminant
        float Lxx = (ptr[-1] - 2 * ptr[0] + ptr[1]);
        if (Lxx < 0)
            return HessianDetector::HESSIAN_DARK;
        else
            return HessianDetector::HESSIAN_BRIGHT;
    }
}

HessianDetectorParams::HessianDetectorParams(int const   inBorderSize,
                                             float const inThreshold,
                                             float const inEdgeEigenValueRatio)
    : Threshold(inThreshold), EdgeEigenValueRatio(inEdgeEigenValueRatio), borderSize(inBorderSize),
      edgeScoreThreshold(std::pow(EdgeEigenValueRatio + 1., 2) / EdgeEigenValueRatio),
      finalThreshold(std::pow(inThreshold, 2)), positiveThreshold(.8 * finalThreshold),
      negativeThreshold(-positiveThreshold)

{
}

HessianDetector::HessianDetector(int const   nOctaves,
                                 int const   nLayers,
                                 int const   nBorders,
                                 float const startSigma,
                                 float const edgeEigvalRatio,
                                 float const threshold)
    : param_pyr({.numOctaves = nOctaves, .numLayers = nLayers, .initialSigma = startSigma}),
      param_detect(nBorders, threshold, edgeEigvalRatio)
{
}

std::vector<CandidatePoint>
HessianDetector::FindOctaveCandidates(HessianResponseOctave const& Octave)
{
    std::vector<CandidatePoint>   Result;
    std::unordered_set<cv::Point> VisitMap;
    for (int LayerIdx = 1; LayerIdx < param_pyr.numLayers - 1; ++LayerIdx)
    {
        auto&& Candidates = FindLayerCandidates(Octave, LayerIdx, VisitMap);
        for (auto& Point : Candidates)
        {
            Point.layer_idx = LayerIdx;
        }
        Result.insert(Result.end(), Candidates.begin(), Candidates.end());
    }

    return Result;
}
std::vector<CandidatePoint>
HessianDetector::FindLayerCandidates(HessianResponseOctave const&   Octave,
                                     int const                      LayerIdx,
                                     std::unordered_set<cv::Point>& VisitMap)
{
    std::vector<CandidatePoint> Result;

    auto const& Prev    = Octave[LayerIdx - 1];
    auto const& Current = Octave[LayerIdx];
    auto const& Next    = Octave[LayerIdx + 1];

    auto const border = param_detect.borderSize;

    auto const rows = Current.rows;
    auto const cols = Current.cols;

    for (int r = border; r < rows - border; ++r)
    {
        auto const* row = Current.ptr<float>(r);
        for (int c = border; c < cols - border; ++c)
        {

            auto const value = row[c];

            bool const BeyondPositiveThreshold = (value > param_detect.positiveThreshold);
            bool const BeyondNegativeThreshold = (value < param_detect.negativeThreshold);

            bool const IsRegionalMaximum = utils::IsRegionMax(Prev, value, r, c) &&
                                           utils::IsRegionMax(Current, value, r, c) &&
                                           utils::IsRegionMax(Next, value, r, c);

            bool const IsRegionalMinimum = utils::IsRegionMin(Prev, value, r, c) &&
                                           utils::IsRegionMin(Current, value, r, c) &&
                                           utils::IsRegionMin(Next, value, r, c);

            if ((BeyondPositiveThreshold && IsRegionalMaximum) ||
                (BeyondNegativeThreshold && IsRegionalMinimum))
            {
                CandidatePoint Point;
                if (LocalizeCandidate(Point,
                                      VisitMap,
                                      Prev,
                                      Current,
                                      Next,
                                      r,
                                      c,
                                      Octave.GetLayerSigma(LayerIdx),
                                      Octave.PixelDistance))
                {
                    auto const* Blur =
                        Octave.GetLayerBlur(LayerIdx).ptr<float>(Point.row) + Point.col;
                    Point.type = GetHessianPointType(Blur, Point.response);
                    Result.emplace_back(Point);
                }
            }
        }
    }
    return Result;
}

bool
HessianDetector::LocalizeCandidate(CandidatePoint&                Point,
                                   std::unordered_set<cv::Point>& VisitMap,
                                   cv::Mat const&                 Prev,    // HessianResponses
                                   cv::Mat const&                 Current, // HessianResponses
                                   cv::Mat const&                 Next,    // HessianResponses
                                   int const                      Row,
                                   int const                      Col,
                                   float const                    CurSigma,
                                   float const                    pixelDistance)
{
    static constexpr auto MaxSubpixelShift  = 0.6;
    static constexpr auto PointSafetyBorder = 2;
    static constexpr auto ScaleThreshold    = 1.5;

    int const Cols = Current.cols;
    int const Rows = Current.rows;

    cv::Mat pixel_shift;

    bool  converged = false;
    int   next_row = Row, next_col = Col;
    int   solution_row, solution_col;
    float Response = 0.;

    auto&& calculate_jacobian = [&Prev, &Current, &Next](
                                    int const r, int const c, cv::Mat& Result) {
        float dxx = at<float>(Current, r, c - 1) - 2.0f * at<float>(Current, r, c) +
                    at<float>(Current, r, c + 1);

        float dyy = at<float>(Current, r - 1, c) - 2.0f * at<float>(Current, r, c) +
                    at<float>(Current, r + 1, c);

        float dss = Prev.at<float>(r, c) - 2.0f * Current.at<float>(r, c) + Next.at<float>(r, c);

        float dxy = 0.25f * (at<float>(Current, r + 1, c + 1) - at<float>(Current, r + 1, c - 1) -
                             at<float>(Current, r - 1, c + 1) + at<float>(Current, r - 1, c - 1));

        float dxs = 0.25f * (at<float>(Next, r, c + 1) - at<float>(Next, r, c - 1) -
                             at<float>(Prev, r, c + 1) + at<float>(Prev, r, c - 1));

        float dys = 0.25f * (at<float>(Next, r + 1, c) - at<float>(Next, r - 1, c) -
                             at<float>(Prev, r + 1, c) + at<float>(Prev, r - 1, c));

        // clang-format off
        auto&& vec = std::array{
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss,
        };
        cv::Mat(3, 3, CV_32F, vec.data()).copyTo(Result);
        // clang-format on
    };

    auto&& calculate_gradient =
        [&Prev, &Current, &Next](int const r, int const c, cv::Mat& Result) {
            float dx = -0.5f * (at<float>(Current, r, c + 1) - at<float>(Current, r, c - 1));
            float dy = -0.5f * (at<float>(Current, r + 1, c) - at<float>(Current, r - 1, c));
            float ds = -0.5f * (at<float>(Next, r, c) - at<float>(Prev, r, c));

            auto&& arr = std::array{dx, dy, ds};

            cv::Mat(3, 1, CV_32F, arr.data()).copyTo(Result);
        };

    {
        float dxx = at<float>(Current, Row, Col - 1) - 2.0f * at<float>(Current, Row, Col) +
                    at<float>(Current, Row, Col + 1);

        float dyy = at<float>(Current, Row - 1, Col) - 2.0f * at<float>(Current, Row, Col) +
                    at<float>(Current, Row + 1, Col);

        float dxy =
            0.25f * (at<float>(Current, Row + 1, Col + 1) - at<float>(Current, Row + 1, Col - 1) -
                     at<float>(Current, Row - 1, Col + 1) + at<float>(Current, Row - 1, Col - 1));

        float edgeScore = (dxx + dyy) * (dxx + dyy) / (dxx * dyy - dxy * dxy);
        if (edgeScore > param_detect.edgeScoreThreshold || edgeScore < 0)
        {
            return false;
        }
    }

    for (int Iter = 0; Iter < 5; ++Iter)
    {
        int const r = next_row;
        int const c = next_col;

        cv::Mat System;
        cv::Mat Constant;
        calculate_jacobian(r, c, System);
        calculate_gradient(r, c, Constant);
        cv::Mat Solution;

        cv::solve(System, Constant, Solution);
        auto const* solution_ptr = Solution.ptr<float>();
        auto const* gradient_ptr = Constant.ptr<float>();

        if (std::isnan(solution_ptr[0]) || std::isnan(solution_ptr[1]) ||
            std::isnan(solution_ptr[2]))
        {
            return false;
        }
        auto const value = Current.at<float>(r, c) + 0.5 * (gradient_ptr[0] * solution_ptr[0] +
                                                            gradient_ptr[1] * solution_ptr[1] +
                                                            gradient_ptr[2] * solution_ptr[2]);

        Solution.copyTo(pixel_shift);
        solution_row = r;
        solution_col = c;
        Response     = value;

        if (solution_ptr[0] > MaxSubpixelShift)
        {
            if (c < Cols - PointSafetyBorder)
            {
                next_col++;
            }
            else
            {
                return false;
            }
        }
        if (solution_ptr[1] > MaxSubpixelShift)
        {
            if (r < Rows - PointSafetyBorder)
            {
                next_row++;
            }
            else
            {
                return false;
            }
        }
        if (solution_ptr[0] < -MaxSubpixelShift)
        {
            if (c > PointSafetyBorder)
            {
                next_col--;
            }
            else
            {
                return false;
            }
        }
        if (solution_ptr[1] < -MaxSubpixelShift)
        {
            if (r > PointSafetyBorder)
                next_row--;
            else
                return false;
        }
        if (next_row == r && next_col == c)
        {
            // converged, displacement is sufficiently small, terminate here
            // TODO: decide if we want only converged local extrema...
            converged = true;
            break;
        }
    }
    auto const* shift_ptr        = pixel_shift.ptr<float>(0);
    bool const  LocalizationTest = std::abs(shift_ptr[0]) > ScaleThreshold ||
                                  std::abs(shift_ptr[1]) > ScaleThreshold ||
                                  std::abs(shift_ptr[2]) > ScaleThreshold;
    bool const ResponseTest   = std::abs(Response) < param_detect.finalThreshold;
    bool const AlreadyVisited = VisitMap.contains(cv::Point(solution_row, solution_col));

    // if spatial localization was all right and the scale is close enough...
    if (AlreadyVisited || LocalizationTest || ResponseTest)
    {
        return false;
    }

    // mark we've visited here
    VisitMap.emplace(cv::Point(solution_row, solution_col));

    float s = CurSigma * std::pow(2., shift_ptr[2] / param_pyr.numLayers);

    Point.x              = pixelDistance * (solution_col + shift_ptr[0]);
    Point.y              = pixelDistance * (solution_row + shift_ptr[1]);
    Point.s              = pixelDistance * s;
    Point.response       = Response;
    Point.pixel_distance = pixelDistance;
    Point.row            = next_row;
    Point.col            = next_col;

    return true;
}

std::vector<CandidatePoint>
HessianDetector::DetectCandidates(HessianResponsePyramid const& Pyr)
{
    std::vector<CandidatePoint> Result;
    for (int Octave = 0; Octave < param_pyr.numOctaves; ++Octave)
    {
        auto&& Candidates = FindOctaveCandidates(Pyr[Octave]);

        for (auto& Point : Candidates)
        {
            Point.octave_idx = Octave;
        }
        Result.insert(Result.end(), Candidates.begin(), Candidates.end());
    }

    return Result;
}

} // namespace ha