
#include "HessianPyramid.hpp"
#include "HessianDetector.hpp"

static constexpr auto path =
    "//ttinno/DroneMapResouce/DroneMapSouce/Gyeonggi_Anyang_Pyeongchon/Anyang_16area/16-2_60d/DJI_"
    "202308161553_002_Anyang3D-16-60-v3-123/DJI_20230816163047_0160.JPG";
int
main()
{
    auto&& Img = cv::imread(path);

    cv::Mat Resized;
    cv::resize(Img, Resized, {0, 0}, 1. / 4, 1. / 4, cv::INTER_AREA);

    cv::Mat Actual;

    Resized.convertTo(Actual, CV_32F);

    ha::HessianDetector det(3, 5, 5, 1.6, 10, 5.333);

    auto a = det.DetectCandidates(Actual);

    std::cout << a.size() << '\n';

    ha::HessianResponsePyramid p(Resized, 3, 5, 1.6);
    cv::imshow("w", p[0][4]);

    cv::waitKey(-1);
}