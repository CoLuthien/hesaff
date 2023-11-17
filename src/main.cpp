
#include "HessianPyramid.hpp"
#include "HessianDetector.hpp"
#include "HessianAffineDetector.hpp"
#include "AffineDeformer.hpp"

static constexpr auto path  = "C:/Users/bumdo/workspace/rcautomation/Images/DJI_0316.JPG";
static constexpr auto path2 = "C:/Users/bumdo/workspace/rcautomation/Images/DJI_0317.JPG";
int
main()
{
    auto&&  Img1 = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::Mat Resize1;
    cv::resize(Img1, Resize1, {}, 0.5, 0.5, cv::INTER_AREA);
    cv::Mat Actual;
    Resize1.convertTo(Actual, CV_32F);

    cv::Mat Resize2;
    auto&&  Img2 = cv::imread(path2, cv::IMREAD_GRAYSCALE);
    cv::resize(Img2, Resize2, {}, 0.5, 0.5, cv::INTER_AREA);
    cv::Mat Actual2;
    Resize2.convertTo(Actual2, CV_32F);

    auto matcher = cv::BFMatcher::create();

    ha::HessianAffineDetector adet(cv::SIFT::create(), {});
    {
        cv::Mat Desc1, Desc2;

        std::vector<cv::KeyPoint> kp1{}, kp2{};

        adet.detectAndCompute(Actual, {}, kp1, Desc1);
        adet.detectAndCompute(Actual2, {}, kp2, Desc2);

        std::vector<cv::Point2f> pt1, pt2;
        std::vector<cv::DMatch>  Matches;

        matcher->match(Desc1, Desc2, Matches);
        cv::KeyPoint::convert(kp1, pt1);
        cv::KeyPoint::convert(kp2, pt2);

        std::cout << pt1.size() << '\t' << pt2.size() << '\n';

        cv::Mat Mask;

        cv::findHomography(pt1, pt2, Mask, cv::USAC_FAST);
        std::cout << "hierere\n";

        cv::Mat View;

        cv::Mat View1, View2;

        std::vector<cv::DMatch> FilteredMatches;
        auto const*             ptr = Mask.ptr<uint8_t>();
        for (auto i = 0; i < Mask.cols; ++i)
        {
            if (ptr[i] > 0)
            {
                FilteredMatches.emplace_back(Matches[i]);
            }
        }
        std::cout << "hierere\n";

        cv::drawMatches(Resize1,
                        kp1,
                        Resize2,
                        kp2,
                        FilteredMatches,
                        View,
                        cv::Scalar::all(-1),
                        cv::Scalar::all(-1),
                        {},
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        cv::imshow("W", View);
        cv::waitKey(-1);
    }
}