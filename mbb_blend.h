#ifndef PANORAMASTITCH_MBBBLEND_H
#define PANORAMASTITCH_MBBBLEND_H

#include "pano_blend.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

class MultiBandBlend : public PanoBlend {
public:
    MultiBandBlend(const std::vector<cv::Mat>& masks);
    void Blend(const std::vector<cv::Mat>& images, cv::Mat& result);
    void Blend(const std::vector<cv::cuda::GpuMat>& images, cv::cuda::GpuMat& result);
protected:
    void BlendPyramidLayers();
private:
    int stitch_num, band_num;
    cv::Rect dst_roi;
    cv::Mat dst_mask;
    std::vector<std::vector<cv::Mat> > src_laplace_pyr, src_weight_pyr;
    std::vector<cv::Mat> dst_laplace_pyr, dst_weight_pyr;

    // GPU data
    std::vector<std::vector<cv::cuda::GpuMat> > d_src_laplace_pyr, d_src_weight_pyr;
    std::vector<cv::cuda::GpuMat> d_dst_laplace_pyr, d_dst_weight_pyr;
    cv::cuda::GpuMat d_dst_mask;
};

#endif // PANORAMASTITCH_MBBBLEND_H
