#include "pano_blend.h"

#define BLEND_WEIGHT_EPS 1e-5

/*
 * NaiveBlend
 */
NaiveBlend::NaiveBlend(const std::vector<cv::Mat>& masks) : masks(masks) {
}

void NaiveBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {
    for (int iter = 0; iter < images.size(); iter++) {
        images[iter].copyTo(result, masks[iter]);
    }
}

/*
 * FeatherBlend
 */

FeatherBlend::FeatherBlend(const std::vector<cv::Mat>& masks, float sharpness) : masks(masks) {
    stitch_num = masks.size();
    for (auto& mask : masks) {
        dst_roi.width = std::max(mask.cols, dst_roi.width);
        dst_roi.height = std::max(mask.rows, dst_roi.height);
    }
    int w = dst_roi.width, h = dst_roi.height;
    weights.resize(stitch_num);
    dst_weight = cv::Mat::zeros(dst_roi.size(), CV_32F);
    for (int i = 0; i < stitch_num; i++) {
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

void FeatherBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {
    int w = dst_roi.width, h = dst_roi.height;
    result = cv::Mat::zeros(h, w, CV_32FC3);
    for (size_t iter = 0; iter < stitch_num; iter ++) {
        for (int y = 0; y < h; ++y) {
            const cv::Point3_<uchar>* src_row = images[iter].ptr<cv::Point3_<uchar> >(y);
            const float* weight_row = weights[iter].ptr<float>(y);
            cv::Point3_<float>* dst_row = result.ptr<cv::Point3_<float> >(y);

            for (int x = 0; x < w; ++x) {
                dst_row[x].x += (float)src_row[x].x * weight_row[x];
                dst_row[x].y += (float)src_row[x].y * weight_row[x];
                dst_row[x].z += (float)src_row[x].z * weight_row[x];
            }
        }
    }
    for (int y = 0; y < h; ++y) {
        const float* dst_weight_row = dst_weight.ptr<float>(y);
        cv::Point3_<float>* dst_row = result.ptr<cv::Point3_<float> >(y);

        for (int x = 0; x < w; ++x) {
            dst_row[x].x /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
            dst_row[x].y /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
            dst_row[x].z /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
        }
    }
    result.convertTo(result, CV_8U);
}

/*
* MultiBandBlend
*/
MultiBandBlend::MultiBandBlend(const std::vector<cv::Mat>& masks) {
    stitch_num = masks.size();
    for (auto& mask : masks) {
        dst_roi.width = std::max(mask.cols, dst_roi.width);
        dst_roi.height = std::max(mask.rows, dst_roi.height);
    }
    int w = dst_roi.width, h = dst_roi.height;
    // calculate band number
    int band_w = static_cast<int>(ceil(log((double)w) / log(2.0))), band_h = static_cast<int>(ceil(log((double)h) / log(2.0)));
    band_num = std::max(band_w, band_h);
    dst_roi.width = dst_roi.height = 1 << band_num;

    // intialize memory
    dst_weight_pyr.resize(band_num + 1);
    dst_laplace_pyr.resize(band_num + 1);
    dst_weight_pyr[0] = cv::Mat::zeros(dst_roi.size(), CV_32F);
    dst_laplace_pyr[0] = cv::Mat::zeros(dst_roi.size(), CV_32FC3);
    for (int layer = 0; layer < band_num; layer++) {
        dst_weight_pyr[layer + 1] = cv::Mat::zeros(dst_weight_pyr[layer].size() / 2, CV_32F);
        dst_laplace_pyr[layer + 1] = cv::Mat::zeros(dst_laplace_pyr[layer].size() / 2, CV_32FC3);
    }

    src_weight_pyr.resize(stitch_num);
    src_laplace_pyr.resize(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        src_weight_pyr[iter].resize(band_num + 1);
        src_laplace_pyr[iter].resize(band_num + 1);

        src_weight_pyr[iter][0] = cv::Mat::zeros(dst_roi.size(), CV_32F);
        src_laplace_pyr[iter][0] = cv::Mat::zeros(dst_roi.size(), CV_32FC3);
        for (int layer = 0; layer < band_num; layer++) {
            src_weight_pyr[iter][layer + 1] = cv::Mat::zeros(src_weight_pyr[iter][layer].size() / 2, CV_32F);
            src_laplace_pyr[iter][layer + 1] = cv::Mat::zeros(src_laplace_pyr[iter][layer].size() / 2, CV_32FC3);
        }
    }
    
    for (int iter = 0; iter < stitch_num; iter++) {
        cv::Mat mask_float;
        masks[iter].convertTo(mask_float, CV_32F, 1 / 255.f);
       
        cv::copyMakeBorder(mask_float, src_weight_pyr[iter][0], 0, dst_roi.height - mask_float.rows, 0, dst_roi.width - mask_float.cols, cv::BORDER_REFLECT);
        for (int layer = 0; layer < band_num; layer++) {
            cv::pyrDown(src_weight_pyr[iter][layer], src_weight_pyr[iter][layer + 1]);
        }
        for (int layer = 0; layer <= band_num; layer++) {
            cv::add(src_weight_pyr[iter][layer], dst_weight_pyr[layer], dst_weight_pyr[layer]);
        }
    }
    dst_weight_pyr[0](cv::Rect(0, 0, w, h)).copyTo(dst_mask);
    dst_mask = dst_mask > BLEND_WEIGHT_EPS;
}

void MultiBandBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {
    for (int iter = 0; iter < stitch_num; iter ++) {
        cv::copyMakeBorder(images[iter], src_laplace_pyr[iter][0], 0, dst_roi.height - images[iter].rows, 0, dst_roi.width - images[iter].cols, cv::BORDER_REFLECT);
        //images[iter].copyTo(src_laplace_pyr[iter][0]);
        src_laplace_pyr[iter][0].convertTo(src_laplace_pyr[iter][0], CV_32F);
        for (int layer = 0; layer < band_num; layer++) {
            cv::pyrDown(src_laplace_pyr[iter][layer], src_laplace_pyr[iter][layer + 1]);
            cv::Mat up_layer;
            cv::pyrUp(src_laplace_pyr[iter][layer + 1], up_layer);
            cv::add(src_laplace_pyr[iter][layer], -up_layer, src_laplace_pyr[iter][layer]);
        }
    }
    for (int layer = 0; layer <= band_num; layer++) {
        cv::Mat& dst_image = dst_laplace_pyr[layer];
        dst_image = cv::Mat::zeros(dst_weight_pyr[layer].size(), CV_32FC3);
        int w = dst_image.cols, h = dst_image.rows;
        for (int iter = 0; iter < stitch_num; iter++) {
            const cv::Mat& image = src_laplace_pyr[iter][layer];
            const cv::Mat& weight = src_weight_pyr[iter][layer];
            
            for (int y = 0; y < h; ++y) {
                const cv::Point3_<float>* src_row = image.ptr<cv::Point3_<float> >(y);
                const float* weight_row = weight.ptr<float>(y);
                cv::Point3_<float>* dst_row = dst_image.ptr<cv::Point3_<float> >(y);

                for (int x = 0; x < w; ++x) {
                    dst_row[x].x += (float)src_row[x].x * weight_row[x];
                    dst_row[x].y += (float)src_row[x].y * weight_row[x];
                    dst_row[x].z += (float)src_row[x].z * weight_row[x];
                }
            }
        }
        for (int y = 0; y < h; ++y) {
            const float* dst_weight_row = dst_weight_pyr[layer].ptr<float>(y);
            cv::Point3_<float>* dst_row = dst_laplace_pyr[layer].ptr<cv::Point3_<float> >(y);

            for (int x = 0; x < w; ++x) {
                dst_row[x].x /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
                dst_row[x].y /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
                dst_row[x].z /= dst_weight_row[x] + BLEND_WEIGHT_EPS;
            }
        }
        
    }
    for (int layer = band_num; layer > 0; layer--) {
        cv::Mat up_layer;
        cv::pyrUp(dst_laplace_pyr[layer], up_layer);
        std::cout << up_layer.size() << std::endl;
        cv::add(up_layer, dst_laplace_pyr[layer - 1], dst_laplace_pyr[layer - 1]);
    }
    dst_laplace_pyr[0](cv::Rect(0, 0, dst_mask.cols, dst_mask.rows)).copyTo(result, dst_mask);
    result.convertTo(result, CV_8U);
}

/*
* ConvPyrBlend
*/
ConvPyrBlend::ConvPyrBlend(const std::vector<cv::Mat>& masks) {

}

void ConvPyrBlend::Blend(const std::vector<cv::Mat>& images, cv::Mat& result) {

}