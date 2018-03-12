#include "mvc_blend.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <sstream>

using namespace std;
using namespace cv;
using cv::cuda::GpuMat;

// avoid border clipping in OpenCV findContours
// choose the largest contour
void FindContour(const Mat& img, vector<Point>& contours, int approx = CV_CHAIN_APPROX_NONE) {
    Mat tmp = Mat::zeros(img.rows + 2, img.cols + 2, img.type());
    img.copyTo(tmp(Rect(1, 1, img.cols, img.rows)));
    vector<vector<Point> > _contours;
    vector<Vec4i> hierarchy;
    findContours(tmp, _contours, hierarchy, CV_RETR_EXTERNAL, approx, Point(-1, -1));
    int idx = 0, max = 0;
    for (int i = 0; i < _contours.size(); i++) {
        if (_contours[i].size() > max) {
            max = _contours[i].size();
            idx = i;
        }
    }
    contours = _contours[idx];
}

void ScaledShow(const string& str, const Mat& image, double scale = 0.5) {
    Mat tmp_image = image.clone();
    resize(tmp_image, tmp_image, Size(), scale, scale);
    imshow(str, tmp_image);
    waitKey();
}

void CalculateBoundary(const vector<Mat>& pano_masks, vector<Mat>& blend_masks, vector<vector<Point> >& boundaries, vector<vector<int> >& seams_idx) {
    int pano_h, pano_w;
    int stitch_num = pano_masks.size();
    if (!pano_masks.empty()) {
        pano_h = pano_masks[0].rows;
        pano_w = pano_masks[0].cols;
    }

    // copy to blend masks (size h * 2w)
    blend_masks.resize(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        Mat m = pano_masks[iter].clone();
        int w = m.cols, h = m.rows;
        blend_masks[iter] = Mat::zeros(h, 2 * w, m.type());

        int ca = 0, cb = 0, ctop = 0;
        int xmin = m.cols, xmax = -1;
        for (int i = 0; i < m.rows; i++) {
            if (m.at<uchar>(i, 0) > 0) ca++;
            if (m.at<uchar>(i, m.cols - 1) > 0) cb++;
        }
        for (int i = 0; i < m.cols; i++) {
            if (m.at<uchar>(0, i) > 0) ctop++;
        }
        if (ca > 0 && cb > 0) {
            if (ctop == m.cols) { // sky
                pano_masks[iter].copyTo(blend_masks[iter](Rect(0, 0, w, h)));
            }
            else { // cut
                Rect left(0, 0, w / 2, h), right(w / 2, 0, w / 2, h);
                pano_masks[iter](right).copyTo(blend_masks[iter](right));
                pano_masks[iter](left).copyTo(blend_masks[iter](left + Point(w, 0)));
            }
        }
        else {
            pano_masks[iter].copyTo(blend_masks[iter](Rect(0, 0, w, h)));
        }
    }
    // dilate masks
    int dilation_size = 3;
    Mat element = getStructuringElement(MORPH_ELLIPSE,
        Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        Point(dilation_size, dilation_size));
    for (int i = 0; i < stitch_num; i++)
        dilate(blend_masks[i], blend_masks[i], element);
    // calculate boundary
    for (int iter = 1; iter < stitch_num; iter++) {
        FindContour(blend_masks[iter], boundaries[iter - 1]);
        unique(boundaries[iter - 1].begin(), boundaries[iter - 1].end());
    }
    // calculate seam
    //Mat tmp_mask(pano_h, pano_w, CV_8U);
    Mat tmp_mask = blend_masks[0].clone();
    const int dx[] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    const int dy[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
    for (int i = 1; i < blend_masks.size(); i++) {
        for (int k = 0; k < boundaries[i - 1].size(); k++) {
            Point pt = boundaries[i - 1][k];
            int ty = pt.y, tx = (pt.x) % pano_w, tx2 = pt.x;
            if (ty < 0 || ty >= pano_h || tx2 < 0 || tx2 >= 2 * pano_w) continue;
            if (blend_masks[i].at<uchar>(ty, tx2) > 0 && (tmp_mask.at<uchar>(ty, tx) > 0)) {
                seams_idx[i - 1].push_back(k);
            }
        }
        for (int row = 0; row < pano_h; row++) {
            for (int col = 0; col < pano_w * 2; col++) {
                uchar cur = blend_masks[i].at<uchar>(row, col);
                if (cur == 0) continue;
                tmp_mask.at<uchar>(row, col % pano_w) = 255;
            }
        }
        /*Mat tmp_show = Mat::zeros(pano_h, pano_w, CV_8U);
        for (int k = 0; k < seams_idx[i - 1].size(); k++) {
            int idx = seams_idx[i - 1].at(k);
            Point p = boundaries[i - 1][idx];
            tmp_show.at<uchar>(p.y, p.x % pano_w) = 255;
        }
        ScaledShow("show", tmp_show);*/
    }
    blend_masks.erase(blend_masks.begin());
}

int main() {
    const string example_dir = "example/";
    const string remap_image_dir = example_dir + "remap/";
    const string mask_image_dir = example_dir + "masks/";

    int stitch_num = 6;

    vector<Mat> images(stitch_num), pano_masks(stitch_num);
    for (int iter = 0; iter < stitch_num; iter++) {
        ostringstream str;
        str << remap_image_dir << iter << ".png";
        images[iter] = imread(str.str(), 1);
        str.str(string());
        str << mask_image_dir << "MBB.Mask." << iter << ".png";
        pano_masks[iter] = imread(str.str(), 0);
    }

    int pano_h = pano_masks[0].rows;
    int pano_w = pano_masks[0].cols;
    // calculate boundary
    vector<vector<Point> > boundaries(stitch_num - 1);
    vector<vector<int> > seams_idx(stitch_num - 1);
    vector<Mat> blend_masks;
    CalculateBoundary(pano_masks, blend_masks, boundaries, seams_idx);

    MVCBlend blender(blend_masks, boundaries, seams_idx, pano_w, pano_h);

    Mat result;
    //blender.Blend(images, result);
    // gpu blend
    GpuMat d_image, d_result;
    Mat tmp_image = images[0].clone();
    for (int iter = 1; iter < stitch_num; iter++) {
        vconcat(tmp_image, images[iter], tmp_image);
    }
    d_image.upload(tmp_image);
    blender.Blend(d_image, d_result);
    d_result.download(result);
    cout << result.size() << endl;

    imshow("result", result);
    waitKey();

    return 0;
}