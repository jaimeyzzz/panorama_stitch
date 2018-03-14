#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev.hpp>
#include <iostream>

#include "mbb_blend.h"

using namespace cv::cuda;
using cv::cudev::divUp;
using namespace std;

__global__ void BlendLayer(PtrStepSz<float3> image,
                           PtrStepSz<float> weight,
                           PtrStepSz<float3> dst_image,
                           int blend_w, int blend_h) {

    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int x = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= blend_w || y >= blend_h) {
        return;
    }
    dst_image(y, x).x = dst_image(y, x).x + image(y, x).x * weight(y, x);
    dst_image(y, x).y = dst_image(y, x).y + image(y, x).y * weight(y, x);
    dst_image(y, x).z = dst_image(y, x).z + image(y, x).z * weight(y, x);
}


__global__ void NormalizeLayer(PtrStepSz<float3> dst_image, 
                               PtrStepSz<float> weight,
                               int blend_w, int blend_h) {
    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int x = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= blend_w || y >= blend_h) {
        return;
    }
    dst_image(y, x).x /= (weight(y, x) + BLEND_WEIGHT_EPS);
    dst_image(y, x).y /= (weight(y, x) + BLEND_WEIGHT_EPS);
    dst_image(y, x).z /= (weight(y, x) + BLEND_WEIGHT_EPS);
}

void MultiBandBlend::BlendPyramidLayers() {
    const dim3 thread_size_2d = dim3(1, 512);
    for (int layer = 0; layer <= band_num; layer++) {
        int blend_w = d_dst_laplace_pyr[layer].cols;
        int blend_h = d_dst_laplace_pyr[layer].rows;
        d_dst_laplace_pyr[layer].setTo(cv::Scalar::all(0));

        const dim3 blend_blocks = dim3(divUp(blend_h, thread_size_2d.x), divUp(blend_w, thread_size_2d.y));
        for (int iter = 0; iter < stitch_num; iter++) {
            BlendLayer << < blend_blocks, thread_size_2d >> > (d_src_laplace_pyr[iter][layer],
                                                               d_src_weight_pyr[iter][layer],
                                                               d_dst_laplace_pyr[layer],
                                                               blend_w, blend_h);
            cudaDeviceSynchronize();
        }
        NormalizeLayer << < blend_blocks, thread_size_2d >> > (d_dst_laplace_pyr[layer],
                                                               d_dst_weight_pyr[layer],
                                                               blend_w, blend_h);
        cudaDeviceSynchronize();
    }
}
