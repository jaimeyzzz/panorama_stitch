#ifndef PANORAMASTITCH_PANOBLEND_H
#define PANORAMASTITCH_PANOBLEND_H

#include <opencv2/opencv.hpp>
#include <vector>

class PanoBlend {
public:
    virtual void Blend(const std::vector<cv::Mat>& images, cv::Mat& result) = 0;
};

class NaiveBlend : public PanoBlend {
public:
    NaiveBlend(const std::vector<cv::Mat>& masks);
    void Blend(const std::vector<cv::Mat>& images, cv::Mat& result);
private:
    std::vector<cv::Mat> masks;
};

class FeatherBlend : public PanoBlend {
public:
    FeatherBlend(const std::vector<cv::Mat>& masks, float sharpness = 0.02f);
    void Blend(const std::vector<cv::Mat>& images, cv::Mat& result);
private:
    int stitch_num;
    cv::Rect dst_roi;
    cv::Mat dst_weight;
    std::vector<cv::Mat> masks, weights;
};

class MultiBandBlend : public PanoBlend {
public:
    MultiBandBlend(const std::vector<cv::Mat>& masks);
    void Blend(const std::vector<cv::Mat>& images, cv::Mat& result);
private:
    int stitch_num;
    cv::Rect dst_roi;
    std::vector<std::vector<cv::Mat> > src_laplace_pyr, src_weight_pyr;
    std::vector<cv::Mat> dst_laplace_pyr, dst_weight_pyr;
};

#endif // PANORAMASTITCH_PANOBLEND_H