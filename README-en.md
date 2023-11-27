
# HessianAffineDetector Developer Guide

## Rationale
A Modernized Easy to read and use port of original hessian affine code.


## Windows

1. First for first, we need vcpkg to meet a dependencies. To install vcpkg to your local environment follow the [link](https://vcpkg.io/en/getting-started.html) . Default directory of vcpkg will be "C:/vcpkg". If you install your vcpkg to another directory, you sholud check your CMakeLists.txt and change the directory for it. 

2. ```sh
   # for dependencies
   #at vcpkg installed directory
   ./vcpkg.exe install boost --triplet x64-windows
   ./vcpkg.exe install opencv4[nonfree,contrib,core,cuda,default-features,dnn,jpeg,png,quirc,tiff,webp]:x64-windows
   ```

3. ```sh
   # execute following command to build.
   cd $(HessianAffineDirectory) 
   mkdir build && cd build
   cmake ../ && cmake --build . --config Release --parallel
   ```

## Linux

1. not yet tested

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