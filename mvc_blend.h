#ifndef MVC_MVCBLEND_H
#define MVC_MVCBLEND_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

class MVCBlend {
public:
    MVCBlend(const std::vector<cv::Mat>& masks, const std::vector<std::vector<cv::Point> >& boundary, const std::vector<std::vector<int> >& boundary_idx, int pano_w, int pano_h);
    ~MVCBlend();
    void Blend(const std::vector<cv::Mat>& images, cv::Mat& result);
    void Blend(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result);
    static double AngleBetweenVector(cv::Point v1, cv::Point v2);
    static cv::Vec3d TriangleInterpolation(cv::Point p, cv::Point p1, cv::Point p2, cv::Point p3);
protected:
    void Triangulate(const std::vector<cv::Point>& bound, std::vector<cv::Point>& pts, std::vector<int>& tris, const cv::Mat& mask);
    void ComputeCoords();
    void ComputeCoords(const std::vector<cv::Point>& vertex, const std::vector<cv::Point>& boundary, const std::vector<int>& seam_idx, std::vector<double>& coords);
    void ComputeTriangle();

    void CalculateVertexes(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result_image); // GPU code
    void CalculateFragments(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& result_image); // GPU code
private:
    int pano_w, pano_h;
    // pixel data
    cv::Mat triangle_map;
    cv::Mat triangle_component;
    // boundary data
    std::vector<std::vector<cv::Point> > boundaries; // boundary points
    std::vector<std::vector<int> > seam_elements; // MVC boundary index
    // triangle data
    std::vector<std::vector<cv::Point> > vertexes; // Delaunay triangle vertexes
    std::vector<std::vector<cv::Point> > diff_vertexes; // MVC diff points
    std::vector<std::vector<int> > triangle_elements; // Triangle vertex index
    // mvc data
    std::vector<std::vector<double> > mvc_coords;
    std::vector<std::vector<double> > mvc_diff_coords;
    
    // GPU data
    cv::cuda::GpuMat d_triangle_map;
    cv::cuda::GpuMat d_triangle_component;
    int ** h_boundaries, ** d_boundaries;
    int ** h_seam_elements, ** d_seam_elements;
    int ** h_diff_vertexes, ** d_diff_vertexes;
    int ** h_triangle_elements, ** d_triangle_elements;
    double ** h_mvc_coords, ** h_mvc_diff_coords;
    double ** d_mvc_coords, ** d_mvc_diff_coords;
    // GPU computing
    double ** h_membranes, ** d_membranes;
    double ** h_boundary_diff, ** d_boundary_diff;
};

#endif // MVC_MVCBLEND_H