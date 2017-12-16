#include "pano_remap.h"
#include "pano_blend.h"

#include <opencv2/opencv.hpp>
#include <sstream>

int main() {
    const std::string example_dir = "example/";
    const std::string remap_image_dir = example_dir + "remap/";
    const std::string mask_image_dir = example_dir + "masks/";
    const std::string script_file = example_dir + "script.txt";

    std::vector<cv::Mat> remap_masks;
    PanoRemap pr(script_file);
    remap_masks = pr.get_masks();
    int stitch_num = remap_masks.size();

    std::vector<cv::Mat> images(stitch_num), pano_masks(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        std::ostringstream str;
        str << remap_image_dir << iter << ".png";
        images[iter] = cv::imread(str.str(), 1);
        str.str(std::string());
        str << mask_image_dir << "Pano.Mask." << iter << ".png";
        pano_masks[iter] = cv::imread(str.str(), 0);
    }
    FeatherBlend nb(pano_masks);

    cv::Mat result;
    nb.Blend(images, result);
    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}