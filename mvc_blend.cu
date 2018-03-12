#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudev.hpp>
#include <iostream>

#include "mvc_blend.h"

using namespace cv::cuda;
using cv::cudev::divUp;
using namespace std;

__global__ void CloneZero(const PtrStepSz<uchar3> image,
                          PtrStepSz<uchar3> result_image,
                          int pano_w, int pano_h) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    const int x = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || y < 0 || x >= pano_w || y >= pano_h) {
        return;
    }
    result_image(y, x) = image(y, x);
}

__global__ void CalculateBoundaryDiff(const PtrStepSz<uchar3> image,
                                      PtrStepSz<uchar3> result_image,
                                      int * seam_element,
                                      int * boundary,
                                      double * boundary_diff,
                                      int iter, int seam_size,
                                      int pano_w, int pano_h) {
    const int seam_idx = blockIdx.x * blockDim.x + threadIdx.x; 
    const int point_idx = seam_element[seam_idx];

    if (seam_idx < 0 || seam_idx >= seam_size) {
        return;
    }
    int x = boundary[point_idx * 2];
    int y = boundary[point_idx * 2 + 1];
    uchar3 color_src = image(y + (iter + 1) * pano_h, x % pano_w);
    uchar3 color_dst = result_image(y, x % pano_w);

    double3 diff = make_double3(color_dst.x - color_src.x,
        color_dst.y - color_src.y,
        color_dst.z - color_src.z);
    boundary_diff[seam_idx * 3] = diff.x;
    boundary_diff[seam_idx * 3 + 1] = diff.y;
    boundary_diff[seam_idx * 3 + 2] = diff.z;
}

__global__ void CalculateMembrane(double* mvc_coord,
                                  double* boundary_diff,
                                  double* membrane,
                                  int seam_size, int vertex_size) {
    const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_idx < 0 || vertex_idx >= vertex_size) { 
        return;
    }
    double3 offset = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < seam_size; i++) {
        offset.x += mvc_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3];
        offset.y += mvc_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3 + 1];
        offset.z += mvc_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3 + 2];
    }
    membrane[vertex_idx * 3] = offset.x;
    membrane[vertex_idx * 3 + 1] = offset.y;
    membrane[vertex_idx * 3 + 2] = offset.z;
}

__global__ void CalculateSeamVertex(const PtrStepSz<uchar3> image, 
                                    PtrStepSz<uchar3> result_image,
                                    int * diff_vertex,
                                    double * mvc_diff_coord,
                                    double * boundary_diff,
                                    int iter, int seam_size, int vertex_size,
                                    int pano_w, int pano_h) {
    const int vertex_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex_idx < 0 || vertex_idx >= vertex_size) {
        return;
    }
    const int x = diff_vertex[vertex_idx * 2];
    const int y = diff_vertex[vertex_idx * 2 + 1];

    double3 offset = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < seam_size; i++) {
        offset.x += mvc_diff_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3];
        offset.y += mvc_diff_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3 + 1];
        offset.z += mvc_diff_coord[vertex_idx * seam_size + i] * boundary_diff[i * 3 + 2];
    }
    uchar3 color_src = image(y + (iter + 1) * pano_h, x % pano_w);
    uchar3 result_color;
    for (int k = 0; k < 3; k++) {
        result_color.x = uchar(MAX(MIN((color_src.x + offset.x), 255.0), 0.0));
        result_color.y = uchar(MAX(MIN((color_src.y + offset.y), 255.0), 0.0));
        result_color.z = uchar(MAX(MIN((color_src.z + offset.z), 255.0), 0.0));
    }
    result_image(y, x % pano_w) = result_color;
}

__global__ void CalculateColors(const PtrStepSz<uchar3> image,
                                PtrStepSz<uchar3> result_image,
                                const PtrStepSz<int2> triangle_map,
                                const PtrStepSz<double3> triangle_component,
                                double ** membranes,
                                int ** triangle_elements,
                                int pano_w, int pano_h) {

    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int x = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < 0 || y < 0 || x >= pano_w || y >= pano_h) {
        return;
    }
    int2 info = triangle_map(y, x);
    int image_idx = info.x;
    int triangle_idx = info.y;
    if (image_idx <= 0) return;
    double3 v = triangle_component(y, x);

    const double * membrane = membranes[image_idx - 1];
    const int * triangle = triangle_elements[image_idx - 1];

    double3 color = make_double3(0.0, 0.0, 0.0);
    int p1 = triangle[triangle_idx];
    int p2 = triangle[triangle_idx + 1];
    int p3 = triangle[triangle_idx + 2];
    color.x = membrane[p1 * 3] * v.x + \
              membrane[p2 * 3] * v.y + \
              membrane[p3 * 3] * v.z;
              
    color.y = membrane[p1 * 3 + 1] * v.x + \
              membrane[p2 * 3 + 1] * v.y + \
              membrane[p3 * 3 + 1] * v.z;
    
    color.z = membrane[p1 * 3 + 2] * v.x + \
              membrane[p2 * 3 + 2] * v.y + \
              membrane[p3 * 3 + 2] * v.z;

    uchar3 color_src = image(y + image_idx * pano_h, x);
    uchar3 result_color;
    result_color.x = uchar(MAX(MIN((color_src.x + color.x), 255.0), 0.0));
    result_color.y = uchar(MAX(MIN((color_src.y + color.y), 255.0), 0.0));
    result_color.z = uchar(MAX(MIN((color_src.z + color.z), 255.0), 0.0));
    result_image(y, x) = result_color;
}

void MVCBlend::CalculateVertexes(const GpuMat& image, GpuMat& result_image) {
    cout << pano_h << ' ' << pano_w << endl;
    const uint thread_size = 512;
    const dim3 thread_size_2d = dim3(1, 512);

    int blend_num = boundaries.size();
    // precompute
    result_image.create(cv::Size(pano_w, pano_h), CV_8UC3);
    const dim3 clone_blocks = dim3(divUp(pano_h, thread_size_2d.x), divUp(pano_w, thread_size_2d.y));
    CloneZero << < clone_blocks, thread_size_2d >> > (image, result_image, pano_w, pano_h);

    for (int iter = 0; iter < blend_num; iter++) {
        int vertex_size = vertexes[iter].size();
        int seam_size = seam_elements[iter].size();

        const uint boundary_blocks = divUp(seam_size, thread_size);
        CalculateBoundaryDiff << < boundary_blocks, thread_size >> > (image, result_image,
                                                                      h_seam_elements[iter], 
                                                                      h_boundaries[iter],
                                                                      h_boundary_diff[iter],
                                                                      iter, seam_size,
                                                                      pano_w, pano_h);

        const uint membrane_blocks = divUp(vertex_size, thread_size);
        CalculateMembrane << < membrane_blocks, thread_size >> > (h_mvc_coords[iter], 
                                                                  h_boundary_diff[iter],
                                                                  h_membranes[iter],
                                                                  seam_size, vertex_size);

        if (iter < blend_num - 1) {
            int diff_vertex_size = diff_vertexes[iter].size();
            const uint seam_vertex_blocks = divUp(diff_vertex_size, thread_size);
            CalculateSeamVertex << < seam_vertex_blocks, thread_size >> > (image, result_image,
                                                                           h_diff_vertexes[iter],
                                                                           h_mvc_diff_coords[iter],
                                                                           h_boundary_diff[iter],
                                                                           iter, seam_size, diff_vertex_size,
                                                                           pano_w, pano_h);
        }
    }
}

void MVCBlend::CalculateFragments(const GpuMat & image, GpuMat & result_image) {
    const dim3 thread_size_2d = dim3(1, 512);

    const dim3 color_blocks = dim3(divUp(pano_h, thread_size_2d.x), divUp(pano_w, thread_size_2d.y));
    CalculateColors << < color_blocks, thread_size_2d >> > (image, result_image,
                                                            d_triangle_map, d_triangle_component,
                                                            d_membranes, d_triangle_elements,
                                                            pano_w, pano_h);
}