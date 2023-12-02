
#include "HessianPyramid.hpp"
#include "HessianDetector.hpp"
#include "Utils.hpp"

#include <numbers>

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

static constexpr auto MaxSubpixelShift  = 0.6;
static constexpr auto PointSafetyBorder = 2;
static constexpr auto ScaleThreshold    = 1.5;
auto
caculate_hessian(
    cv::Mat const& Prev, cv::Mat const& Current, cv::Mat const& Next, int const r, int const c)
{
    float dxx = at<float>(Current, r, c - 1) - 2.0f * at<float>(Current, r, c) +
                at<float>(Current, r, c + 1);

    float dyy = at<float>(Current, r - 1, c) - 2.0f * at<float>(Current, r, c) +
                at<float>(Current, r + 1, c);

    float dss = at<float>(Prev, r, c) - 2.0f * at<float>(Current, r, c) + at<float>(Next, r, c);

    float dxy = 0.25f * (at<float>(Current, r + 1, c + 1) - at<float>(Current, r + 1, c - 1) -
                         at<float>(Current, r - 1, c + 1) + at<float>(Current, r - 1, c - 1));

    float dxs = 0.25f * (at<float>(Next, r, c + 1) - at<float>(Next, r, c - 1) -
                         at<float>(Prev, r, c + 1) + at<float>(Prev, r, c - 1));

    float dys = 0.25f * (at<float>(Next, r + 1, c) - at<float>(Next, r - 1, c) -
                         at<float>(Prev, r + 1, c) + at<float>(Prev, r - 1, c));

    return cv::Mat((cv::Mat_<float>(3, 3) << dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss));
};

auto
calculate_gradient(
    cv::Mat const& Prev, cv::Mat const& Current, cv::Mat const& Next, int const r, int const c)
{
    float dx = -0.5f * (at<float>(Current, r, c + 1) - at<float>(Current, r, c - 1));
    float dy = -0.5f * (at<float>(Current, r + 1, c) - at<float>(Current, r - 1, c));
    float ds = -0.5f * (at<float>(Next, r, c) - at<float>(Prev, r, c));
    return cv::Mat((cv::Mat_<float>(3, 1) << dx, dy, ds));
};

auto
TryUpdatePosition(float const Shift, int const current_location, int& next_location, int Space)
    -> bool
{
    if (Shift > MaxSubpixelShift)
    {
        if (current_location < Space - PointSafetyBorder)
        {
            next_location++;
        }
        else
        {
            return false;
        }
    }
    if (Shift < -MaxSubpixelShift)
    {
        if (current_location > PointSafetyBorder)
        {
            next_location--;
        }
        else
        {
            return false;
        }
    }
    return true;
};
HessianDetectorParams::HessianDetectorParams(int const   inBorderSize,
                                             float const inThreshold,
                                             float const inEdgeEigenValueRatio)
    : Threshold(inThreshold), EdgeEigenValueRatio(inEdgeEigenValueRatio), borderSize(inBorderSize),
      edgeScoreThreshold(std::pow(EdgeEigenValueRatio + 1., 2) / EdgeEigenValueRatio),
      finalThreshold(std::pow(inThreshold, 2)), positiveThreshold(.8 * finalThreshold),
      negativeThreshold(-positiveThreshold)

{
}
HessianDetector::HessianDetector(HessianDetectorParams Params)
    : HessianDetector(Params.borderSize, Params.EdgeEigenValueRatio, Params.Threshold)
{
}
HessianDetector::HessianDetector(int const   nBorders,
                                 float const edgeEigvalRatio,
                                 float const threshold)
    : param_detect(nBorders, threshold, edgeEigvalRatio)
{
}

std::vector<CandidatePoint>
HessianDetector::FindOctaveCandidates(HessianResponsePyramid const& Pyr,
                                      std::size_t const             Octave) const
{
    std::vector<CandidatePoint>   Result;
    std::unordered_set<cv::Point> VisitMap;

    auto const numLayers = Pyr.numLayers() - 1;

    for (int LayerIdx = 1; LayerIdx < numLayers; ++LayerIdx)
    {
        auto&& Candidates = FindLayerCandidates(Pyr, Octave, LayerIdx, VisitMap);
        Result.insert(Result.end(), Candidates.begin(), Candidates.end());
    }

    return Result;
}
std::vector<CandidatePoint>
HessianDetector::FindLayerCandidates(HessianResponsePyramid const&  Pyr,
                                     std::size_t const              OctaveIdx,
                                     std::size_t const              LayerIdx,
                                     std::unordered_set<cv::Point>& VisitMap) const
{
    std::vector<CandidatePoint> Result;

    auto const& CurrentOctave = Pyr[OctaveIdx];

    auto const& Prev    = CurrentOctave[LayerIdx - 1];
    auto const& Current = CurrentOctave[LayerIdx];
    auto const& Next    = CurrentOctave[LayerIdx + 1];

    auto const Border = param_detect.borderSize;

    auto const Rows = Current.rows;
    auto const Cols = Current.cols;

    auto const RowIter = Rows - Border;
    auto const ColIter = Cols - Border;

    auto const RowIter = Rows - Border;
    auto const ColIter = Cols - Border;

    auto const LayerSigma    = CurrentOctave.GetLayerSigma(LayerIdx);
    auto const LayerDistance = CurrentOctave.PixelDistance;
    auto const LayerCount    = Pyr.numLayers();

    for (int r = Border; r < RowIter; ++r)
    {
        auto const* row = Current.ptr<float>(r);
        for (int c = Border; c < ColIter; ++c)
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

            if ((BeyondPositiveThreshold & IsRegionalMaximum) |
                (BeyondNegativeThreshold & IsRegionalMinimum))
            {
                CandidatePoint Point;

                if (LocalizeCandidate(Point,
                                      VisitMap,
                                      Prev,
                                      Current,
                                      Next,
                                      r,
                                      c,
                                      LayerSigma,
                                      LayerDistance,
                                      LayerCount))
                {
                    // auto const* Blur =
                    // CurrentOctave.GetLayerBlur(LayerIdx).ptr<float>(Point.y_pos) + Point.x_pos;
                    // Point.type       = GetHessianPointType(Blur, Point.response);// unused remove
                    // for performance
                    Point.octave_idx = OctaveIdx;
                    Point.layer_idx  = LayerIdx;
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
                                   int const                      PositionY,
                                   int const                      PositionX,
                                   float const                    CurSigma,
                                   float const                    pixelDistance,
                                   int const                      NumLayers) const
{

    int const Cols = Current.cols;
    int const Rows = Current.rows;

    cv::Mat pixel_shift;

    bool  converged = false;
    int   NextRow = PositionY, NextCol = PositionX;
    int   solution_row, solution_col;
    float Response = 0.;

    {
        float dxx = at<float>(Current, PositionY, PositionX - 1) -
                    2.0f * at<float>(Current, PositionY, PositionX) +
                    at<float>(Current, PositionY, PositionX + 1);

        float dyy = at<float>(Current, PositionY - 1, PositionX) -
                    2.0f * at<float>(Current, PositionY, PositionX) +
                    at<float>(Current, PositionY + 1, PositionX);

        float dxy = 0.25f * (at<float>(Current, PositionY + 1, PositionX + 1) -
                             at<float>(Current, PositionY + 1, PositionX - 1) -
                             at<float>(Current, PositionY - 1, PositionX + 1) +
                             at<float>(Current, PositionY - 1, PositionX - 1));

        float edgeScore = std::pow(dxx + dyy, 2) / (dxx * dyy - dxy * dxy);
        if (edgeScore > param_detect.edgeScoreThreshold || edgeScore < 0)
        {
            return false;
        }
    }

    for (int Iter = 0; Iter < 5; ++Iter)
    {
        int const r = NextRow;
        int const c = NextCol;

        cv::Mat Hessian  = caculate_hessian(Prev, Current, Next, r, c);
        cv::Mat Gradient = calculate_gradient(Prev, Current, Next, r, c);
        cv::Mat Solution;

        bool Success = cv::solve(Hessian, Gradient, Solution);

        if (!Success)
        {
            return false;
        }
        auto const value = at<float>(Current, r, c) + 0.5 * Solution.dot(Gradient);

        Solution.copyTo(pixel_shift);
        solution_row = r;
        solution_col = c;
        Response     = value;

        auto const* solution_ptr = reinterpret_cast<float*>(Solution.data);

        if (!TryUpdatePosition(solution_ptr[0], c, NextCol, Cols) |
            !TryUpdatePosition(solution_ptr[1], r, NextRow, Rows))
        {
            return false;
        }

        if (NextRow == r && NextCol == c)
        {
            // converged, displacement is sufficiently small, terminate here
            // TODO: decide if we want only converged local extrema...
            converged = true;
            break;
        }
    }
    auto const* shift_ptr        = reinterpret_cast<float*>(pixel_shift.data);
    bool const  LocalizationTest = std::abs(shift_ptr[0]) > ScaleThreshold ||
                                  std::abs(shift_ptr[1]) > ScaleThreshold ||
                                  std::abs(shift_ptr[2]) > ScaleThreshold;
    bool const ResponseTest   = std::abs(Response) < param_detect.finalThreshold;
    bool const AlreadyVisited = VisitMap.contains(cv::Point(solution_col, solution_row));

    // if spatial localization was all right and the scale is close enough...
    if (AlreadyVisited | LocalizationTest | ResponseTest)
    {
        return false;
    }

    // mark we've visited here
    VisitMap.emplace(cv::Point(solution_col, solution_row));

    float scale = CurSigma * std::pow(2., shift_ptr[2] / NumLayers);

    Point = {
        .x              = pixelDistance * (solution_col + shift_ptr[0]),
        .y              = pixelDistance * (solution_row + shift_ptr[1]),
        .s              = pixelDistance * scale,
        .response       = Response,
        .pixel_distance = pixelDistance,
        .orientation    = -1.,
        .y_pos          = NextRow,
        .x_pos          = NextCol,
    };

    return true;
}

std::vector<CandidatePoint>
HessianDetector::DetectCandidates(HessianResponsePyramid const& Pyr) const
{
    std::vector<CandidatePoint> Result;
    auto const                  numOctaves = Pyr.numOctaves();
    for (int OctaveIdx = 0; OctaveIdx != numOctaves; ++OctaveIdx)
    {
        auto&& Candidates = FindOctaveCandidates(Pyr, OctaveIdx);
        Result.insert(Result.end(), Candidates.begin(), Candidates.end());
    }

    return Result;
}

} // namespace ha