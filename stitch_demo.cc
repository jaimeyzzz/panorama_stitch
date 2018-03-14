#include "pano_remap.h"
#include "mbb_blend.h"

#include <opencv2/opencv.hpp>
#include <sstream>

using namespace std;
using cv::Mat;
using cv::cuda::GpuMat;

int main() {
    const string example_dir = "example/";
    const string remap_image_dir = example_dir + "remap/";
    const string mask_image_dir = example_dir + "masks/";
    const string script_file = example_dir + "script.txt";

    vector<Mat> remap_masks;
    PanoRemap pr(script_file);
    remap_masks = pr.get_masks();
    int stitch_num = remap_masks.size();

    vector<Mat> images(stitch_num), pano_masks(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        ostringstream str;
        str << remap_image_dir << iter << ".png";
        images[iter] = cv::imread(str.str(), 1);
        str.str(std::string());
        str << mask_image_dir << "MBB.Mask." << iter << ".png";
        pano_masks[iter] = cv::imread(str.str(), 0);
    }
    MultiBandBlend nb(pano_masks);
    vector<GpuMat> d_images(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        d_images[iter].upload(images[iter]);
    }
    GpuMat d_result;

    Mat result;
    //nb.Blend(images, result);
    nb.Blend(d_images, d_result);
    d_result.download(result);
    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}