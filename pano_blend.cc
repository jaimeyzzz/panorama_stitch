#include "pano_blend.h"

#define WEIGHT_EPS 1e-5

/*
 * NaiveBlend
 */
NaiveBlend::NaiveBlend(const std::vector<cv::Mat>& masks) : masks(masks) {

}

void NaiveBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {
    for (size_t iter; iter < images.size(); iter++) {
        images[iter].copyTo(result, masks[iter]);
    }
}

FeatherBlend::FeatherBlend(const std::vector<cv::Mat>& masks, float sharpness) : masks(masks) {
    num_stitch = masks.size();
    for (auto& mask : masks) {
        dst_roi.width = std::max(mask.cols, dst_roi.width);
        dst_roi.height = std::max(mask.rows, dst_roi.height);
    }
    int w = dst_roi.width, h = dst_roi.height;
    weights.resize(num_stitch);
    dst_weight = cv::Mat::zeros(dst_roi.size(), CV_32F);
    for (int i = 0; i < num_stitch; i++) {
        distanceTransform(masks[i], weights[i], CV_DIST_L1, 3);
        threshold(weights[i] * sharpness, weights[i], 1.f, 1.f, cv::THRESH_TRUNC);
        for (int y = 0; y < h; ++y) {
            const float* weight_row = weights[i].ptr<float>(y);
            float* dst_weight_row = dst_weight.ptr<float>(y);
            for (int x = 0; x < w; ++x) {
                dst_weight_row[x] += weight_row[x];
            }
        }
    }
}

/*
* FeatherBlend
*/
void FeatherBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {
    int w = dst_roi.width, h = dst_roi.height;
    result = cv::Mat::zeros(h, w, CV_32FC3);
    for (size_t iter = 0; iter < num_stitch; iter ++) {
        for (int y = 0; y < h; ++y) {
            const cv::Vec3b* src_row = images[iter].ptr<cv::Vec3b>(y);
            const float* weight_row = weights[iter].ptr<float>(y);
            cv::Vec3b* dst_row = result.ptr<cv::Vec3b>(y);
            float* dstWeightRow = dst_weight.ptr<float>(y);

            for (int x = 0; x < w; ++x) {
                for (int k = 0; k < 3; ++k) {
                    dst_row[x][k] += (float)src_row[x][k] * weight_row[x];
                }
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        const float* dstWeightRow = dst_weight.ptr<float>(y);
        cv::Vec3b* dst_row = result.ptr<cv::Vec3b>(y);

        for (int x = 0; x < w; ++x) {
            for (int k = 0; k < 3; ++k) {
                dst_row[x][k] /= dstWeightRow[x] + WEIGHT_EPS;
            }
        }
    }
    result.convertTo(result, CV_8U);
}

/*
* MultiBandBlend
*/

#undef WEIGHT_EPS