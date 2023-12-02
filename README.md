# HessianAffineDetector Developer Guide

## Rationale

아직 박사님의 코드가 다 해독된게 아니기 때문에 자세한 설명은 생략하겠습니다 ><

## Windows

1. 일단 우리는 vcpkg가 필요합니다. [링크](https://vcpkg.io/en/getting-started.html)를 따라서 vcpkg를 원하는 디렉토리에 설치하세요. 기본 디렉토리는 "C:/vcpkg" 로 설정했습니다. 변경시 CMakeLists.txt에서 vcpkg 패스를 수정하셔야합니다. 

2.  ```sh
    # 의존성 빌드를 위하여 다음 명령어를 실행하세요.
    #at vcpkg installed directory
    ./vcpkg.exe install boost --triplet x64-windows
    ./vcpkg.exe install opencv4[nonfree,contrib,core,cuda,default-features,dnn,jpeg,png,quirc,tiff,webp,tbb,python]:x64-windows
    ```

3. ```sh
   # 다음 명령어를 실행하면 빌드가 되도록 설정했습니다. 안될시 garam.udev@gmail.com으로 연락주세요
   cd $(HessianAffineDirectory) 
   mkdir build && cd build
   cmake ../ && cmake --build . --config Release --parallel
   ```

## Linux

1. 아직 안됐습니다. :( 

## Usage

```cpp

#include "sift_extractor.hpp"
#include "HessianPyramid.hpp"
#include "HessianDetector.hpp"
#include "HessianAffineDetector.hpp"
#include "AffineDeformer.hpp"
#include <filesystem>

auto path  = "PathToFile1.JPG";
auto path2 = "PathToFile2.JPG"
class SURFExtractor : public ha::FeatureExtractor
{
public:
    SURFExtractor() { m_backend = cv::xfeatures2d::SURF::create(100, 4, 3, true); }

public:
    virtual void compute(cv::InputArray             Image,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray            descriptors) override
    {
        m_backend->compute(Image, keypoints, descriptors);
    }

private:
    cv::Ptr<cv::Feature2D> m_backend;
};

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

    ha::HessianAffineDetector adet(std::make_shared<SURFExtractor>());
    {
        cv::Mat Desc1, Desc2;

        std::vector<cv::KeyPoint> kp1{}, kp2{};

        adet.detectAndCompute(Actual, {}, kp1, Desc1);
        adet.detectAndCompute(Actual2, {}, kp2, Desc2);

        std::vector<cv::Point2f>             pt1, pt2;
        std::vector<std::vector<cv::DMatch>> Matches;

        matcher->knnMatch(Desc1, Desc2, Matches, 2);

        std::vector<cv::KeyPoint> v1, v2;
        std::vector<cv::DMatch>   Filtered;
        for (auto& m : Matches)
        {
            auto& m1 = m[0];
            auto& m2 = m[1];

            if (m1.distance < m2.distance * 0.2)
            {
                v1.emplace_back(kp1[m1.queryIdx]);
                v2.emplace_back(kp2[m1.trainIdx]);
                Filtered.emplace_back(m1);
            }
        }
        cv::KeyPoint::convert(v1, pt1);
        cv::KeyPoint::convert(v2, pt2);
        std::cout << pt1.size() << '\t' << pt2.size() << '\n';

        cv::Mat Mask;

        auto&& H = cv::findHomography(pt1, pt2, Mask, cv::USAC_ACCURATE);

        cv::Mat View;

        cv::Mat View1, View2;

        std::cout << Mask.size << '\n';
        cv::drawMatches(Resize1,
                        kp1,
                        Resize2,
                        kp2,
                        Filtered,
                        View,
                        cv::Scalar::all(-1),
                        cv::Scalar::all(-1),
                        Mask,
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        auto path = std::filesystem::current_path().string() + "/out.jpg";
        cv::imwrite(path, View);
    }
}
```



