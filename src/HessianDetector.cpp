
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
    for (int LayerIdx = 1; LayerIdx < Pyr.numLayers() - 1; ++LayerIdx)
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
                                      CurrentOctave.GetLayerSigma(LayerIdx),
                                      CurrentOctave.PixelDistance,
                                      Pyr.numLayers()))
                {
                    auto const* Blur =
                        CurrentOctave.GetLayerBlur(LayerIdx).ptr<float>(Point.y_pos) + Point.x_pos;
                    Point.type       = GetHessianPointType(Blur, Point.response);
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
    static constexpr auto MaxSubpixelShift  = 0.6;
    static constexpr auto PointSafetyBorder = 2;
    static constexpr auto ScaleThreshold    = 1.5;

    int const Cols = Current.cols;
    int const Rows = Current.rows;

    cv::Mat pixel_shift;

    bool  converged = false;
    int   next_row = PositionY, next_col = PositionX;
    int   solution_row, solution_col;
    float Response = 0.;

    auto&& calculate_jacobian = [&Prev, &Current, &Next](int const r, int const c) {
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

        // clang-format off
        auto&& vec = std::array{
            dxx, dxy, dxs,
            dxy, dyy, dys,
            dxs, dys, dss,
        };
        return cv::Mat(3, 3, CV_32F, vec.data()).clone();
        // clang-format on
    };

    auto&& calculate_gradient = [&Prev, &Current, &Next](int const r, int const c) {
        float dx = -0.5f * (at<float>(Current, r, c + 1) - at<float>(Current, r, c - 1));
        float dy = -0.5f * (at<float>(Current, r + 1, c) - at<float>(Current, r - 1, c));
        float ds = -0.5f * (at<float>(Next, r, c) - at<float>(Prev, r, c));

        auto&& arr = std::array{dx, dy, ds};

        return cv::Mat(3, 1, CV_32F, arr.data()).clone();
    };

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

        cv::Mat System   = calculate_jacobian(r, c);
        cv::Mat Constant = calculate_gradient(r, c);
        cv::Mat Solution;

        cv::solve(System, Constant, Solution);
        auto const* solution_ptr = Solution.ptr<float>();
        auto const* gradient_ptr = Constant.ptr<float>();

        if (std::isnan(solution_ptr[0]) || std::isnan(solution_ptr[1]) ||
            std::isnan(solution_ptr[2]))
        {
            return false;
        }
        auto const value = at<float>(Current, r, c) + 0.5 * (gradient_ptr[0] * solution_ptr[0] +
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
    VisitMap.emplace(cv::Point(solution_col, solution_row));

    float scale = CurSigma * std::pow(2., shift_ptr[2] / NumLayers);

    Point.x              = pixelDistance * (solution_col + shift_ptr[0]);
    Point.y              = pixelDistance * (solution_row + shift_ptr[1]);
    Point.s              = pixelDistance * scale;
    Point.response       = Response;
    Point.orientation    = -1.;
    Point.pixel_distance = pixelDistance;
    Point.x_pos          = next_col;
    Point.y_pos          = next_row;

    return true;
}

std::vector<CandidatePoint>
HessianDetector::DetectCandidates(HessianResponsePyramid const& Pyr) const
{
    std::vector<CandidatePoint> Result;
    for (int OctaveIdx = 0; OctaveIdx < Pyr.numOctaves(); ++OctaveIdx)
    {
        auto&& Candidates = FindOctaveCandidates(Pyr, OctaveIdx);
        Result.insert(Result.end(), Candidates.begin(), Candidates.end());
    }

    return Result;
}

} // namespace ha