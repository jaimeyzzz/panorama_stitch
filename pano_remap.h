#ifndef PANORAMASTITCH_PANOREMAP_H_
#define PANORAMASTITCH_PANOREMAP_H_

#include "pano_script.h"

#include <vector>
#include <opencv2/opencv.hpp>

class PanoRemap {
public:
    PanoRemap(std::string);
    ~PanoRemap();

    std::vector<cv::Mat> get_masks();

    void Remap(const std::vector<cv::Mat>& frames, std::vector<cv::Mat>& remap_frames);
private:
    /**
    * @brief initial the remap table for frame remapping
    */
    void initial_remap_table();
    void initial_transforms();

    void Transform(int idx, double, double, double*, double*);

    static double Cubic01(double x);
    static double Cubic12(double x);
private:
    int stitch_num;
    cv::Rect panorama_roi;
    PanoScript ps;
    std::vector<TransformParam> transforms;
    std::vector<cv::Mat> masks;
    double ** remap_table_x, ** remap_table_y;
};

#endif // PANORAMASTITCH_PANOREMAP_H_