#include "PanoStitch.h"
#include "PanoSeamFinder.h"

#include "CmFile.h"
#include <opencv2\opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::detail;

/**********
 static class functions
 */
// show scaled images;
void PanoStitch::showImage(string title, const Mat& img, double scale) {
	Mat tmp;
	img.copyTo(tmp);
	resize(tmp, tmp, Size(), scale, scale);
	imshow(title, tmp);
	waitKey();
}
// fix Opencv::findContours image-border clipping bug.
void PanoStitch::panoFindContours(const Mat & img, vector<Point>& contours, int approx) {
	contours.clear();
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

/**********
 class member functions
 */
PanoStitch::PanoStitch(string scriptName) : ps(scriptName) {
	srand(time(0));
	imNum = ps.images.size();
	w = ps.pano.w, h = ps.pano.h;
	dstRoi.width = w, dstRoi.height = h;
}

PanoStitch::~PanoStitch() {
}

/** set the remap table for frame remapping */
void PanoStitch::initMap() {
	masks.resize(imNum);
	for (int i = 0; i < imNum; i++) {
		masks[i] = Mat::zeros(ps.pano.h, ps.pano.w, CV_8U);
	}

	std::cout << "Set the frame remapping table." << endl;
	//double w2 = (double)ps.pano.w / 2.0 - 0.5, h2 = (double)ps.pano.h / 2.0 - 0.5;

	//mapX.create(dstRoi.height, dstRoi.width * imNum, CV_32F);
	//mapY.create(dstRoi.height, dstRoi.width * imNum, CV_32F);

	//double dx, dy;
	//int iw, ih;
	//for (int idx = 0; idx < imNum; idx++) {
	//	ps.setTransPram(idx);
	//	iw = ps.images[idx].w, ih = ps.images[idx].h;
	//	double sw2 = (double)ps.images[idx].w / 2.0 - 0.5, sh2 = (double)ps.images[idx].h / 2.0 - 0.5;

	//	for (int y = 0; y < dstRoi.height; y++) {
	//		for (int x = 0; x < dstRoi.width; x++) {
	//			int y_ref = y, x_ref = x;
	//			if (y >= ps.pano.h) y_ref = 2 * ps.pano.h - 1 - y;
	//			if (x >= ps.pano.w) x_ref = 2 * ps.pano.w - 1 - x;

	//			double y_d = (double)y_ref - h2;
	//			double x_d = (double)x_ref - w2;
	//			/*double y_d = (double)y - h2;
	//			double x_d = (double)x - w2;*/
	//			ps.transform(x_d, y_d, &dx, &dy);
	//			dx += sw2;
	//			dy += sh2;
	//			float fxc = (float)dx, fyc = (float)dy;
	//			if ((fxc < (float)iw) && (fxc >= 0) && (fyc < (float)ih) && (fyc >= 0)) {
	//				mapX.at<float>(y, x + idx * dstRoi.width) = fxc;
	//				mapY.at<float>(y, x + idx * dstRoi.width) = fyc;

	//				if (y < ps.pano.h && x < ps.pano.w)
	//					masks[idx].at<uchar>(y, x) = 255;
	//			}
	//			else {
	//				int xs = (int)floor(abs(dx) + 0.5);
	//				int ys = (int)floor(abs(dy) + 0.5);
	//				dx = abs(dx) - xs;
	//				dy = abs(dy) - ys;

	//				int xc = xs % (2 * iw), yc = ys % (2 * ih);
	//				if (xc >= iw) {
	//					xc = 2 * iw - 1 - xc;
	//					dx = -dx;
	//				}
	//				if (yc >= ih) {
	//					yc = 2 * ih - 1 - yc;
	//					dy = -dy;
	//				}
	//				// set xc, yc < 0;
	//				mapX.at<float>(y, x + idx * dstRoi.width) = -(dx + (float)xc);
	//				mapY.at<float>(y, x + idx * dstRoi.width) = -(dy + (float)yc);
	//			}
	//		}
	//	}
	//}
}

/** set the order to blending frames */
void PanoStitch::initOrder() {
	if (videoOrder.size() == 0)
		videoOrder = { 0, 1, 2, 3, 4, 5 };
	assert(videoOrder.size() == imNum);
	for (int i = 0; i < videoOrder.size(); i++) {
		if (videoOrder[i] < 0 || videoOrder[i] >= imNum)  {
			cout << "Initial error : wrong orders" << endl;
			system("pause"); exit(0);
		}
	}
	cout << "Set the orders for MVC blending." << endl;
	bool full_ceil = false;
	// calculate ranges;
	for (int iter = 0; iter < videoOrder.size(); iter++) {
		Order ord;
		ord.index = videoOrder[iter];

		Mat& m = masks[ord.index];
		w = m.cols, h = m.rows;

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
			if (ctop == m.cols) {
				xmin = 0, xmax = m.cols - 1, ord.pos = FULL_CEIL; // full ceiling;
				full_ceil = true;
			}
			else {
				for (int i = 0; i < m.rows; i++) {
					for (int j = 0; j < m.cols; j++) {
						if (m.at<uchar>(i, j) == 0) continue;
						int x;
						if (j < m.cols / 2) x = j + m.cols;
						else x = j;
						if (x > xmax) xmax = x;
						if (x < xmin) xmin = x;
					}
				}
				ord.pos = CUT;
			}
		}
		else {
			ord.pos = FULL;
			for (int i = 0; i < m.rows; i++) {
				for (int j = 0; j < m.cols; j++) {
					if (m.at<uchar>(i, j) == 0) continue;
					if (j > xmax) xmax = j;
					if (j < xmin) xmin = j;
				}
			}
		}
		//cout << ord.pos << endl;
		ord.min = xmin; ord.max = xmax;
		orders.push_back(ord);
	}
}

void PanoStitch::copyInOrder(const vector<Mat> &src, vector<Mat> &dest, int type) {
	for (int i = 0; i < orders.size(); i++) {
		Order& ord = orders[i];
		dest.push_back(src[ord.index]);
	}
	// MVC copyInOder
	/*for (int i = 0; i < orders.size(); i++) {
		Order& ord = orders[i];
		Mat mask = Mat::zeros(h, w * 2, type);
		switch (ord.pos) {
		case FULL:
		case FULL_CEIL:
			src[ord.index].copyTo(mask(Rect(0, 0, w, h)));
			break;
		case CUT:
		{
			Rect left(0, 0, w / 2, h), right(w / 2, 0, w / 2, h);
			src[ord.index](right).copyTo(mask(right));
			src[ord.index](left).copyTo(mask(left + Point(w, 0)));
			break;
		}
		default:
			break;
		}
		dest.push_back(mask);
	}*/
}

void PanoStitch::loadVideo(string outdir, string format) {
	videos.resize(imNum);
	vector<string> videoNames;
	CmFile::GetNames(outdir + "*\." + format, videoNames);
	assert(videoNames.size() > 0);
	sort(videoNames.begin(), videoNames.end());

	for (int i = 0; i < imNum; i++) {
		//cout << videoNames[i] << " " << i << endl;
		if (!videos[i].open(outdir + videoNames[i])) {
			std::cout << "Load video error : " << outdir + videoNames[i] << " ?" << endl;
			system("pause");
			exit(0);
		}
	}
	// adjust frames
	if (videoDelay.size() == 0) return;
	/*auto smallestDelay = min_element(videoDelay.begin(), videoDelay.end());
	for (int i = 0; i < videoDelay.size(); i++) {
		videoDelay[i] -= *smallestDelay;
	}
	Mat frame;
	for (int i = 0; i < videos.size(); i++) {
		for (int j = 0; j < videoDelay[i]; j++)
			videos[i].read(frame);
	}*/
}

// Cubic polynomial with parameter A
// A = -1: sharpen; A = - 0.5 homogeneous
// make sure x >= 0
#define	A	(-0.75)
// 0 <= x < 1
inline float PanoStitch::cubic01(float x) {
	return	((A + 2.0)*x - (A + 3.0))*x*x + 1.0;
}
// 1 <= x < 2
inline float PanoStitch::cubic12(float x) {
	return	((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A;
}
#undef A

#define		CUBIC( x, a, NDIM )									\
	a[3] = cubic12(2.0 - x);									\
	a[2] = cubic01(1.0 - x);									\
	a[1] = cubic01(x);											\
	a[0] = cubic12(x + 1.0);									\

void PanoStitch::remap(vector<Mat>& frames, vector<Mat>& remapFrames) {
	remapFrames.resize(imNum);
	for (int i = 0; i < imNum; i++) {
		remapFrames[i] = Mat::zeros(dstRoi.size(), CV_8UC3);
	}
	int wd = dstRoi.width, hd = dstRoi.height;
	for (int idx = 0; idx < imNum; idx++) { // idx : image iterator
		int ws = frames[idx].cols, hs = frames[idx].rows;
		for (int y = 0; y < dstRoi.height; y++) {
			for (int x = 0; x < dstRoi.width; x++) {
				float xx = mapX.at<float>(y, x + idx * wd), yy = mapY.at<float>(y, x + idx * wd);
				if (xx < 0 || y < 0) {
					xx = abs(xx), yy = abs(yy);
					int xc = max(0, min((int)floor(xx), ws - 1)), yc = max(0, min((int)floor(yy), hs - 1));
					remapFrames[idx].at<Vec3b>(y, x) = frames[idx].at<Vec3b>(yc, xx);
					continue;
				}
				// interpolation
				int xc = (int)floor(xx), yc = (int)floor(yy);
				float dx = xx - (float)xc, dy = yy - (float)yc;
				
				int n = 4;
				int n2 = n / 2;

				int ys = yc + 1 - n2; // smallest y-index used for interpolation
				int xs = xc + 1 - n2; // smallest x-index used for interpolation

				float xw[4], yw[4];
				CUBIC(dx, xw, 4);
				CUBIC(dy, yw, 4);
				float weightY = EPS;
				Vec3f sum(0, 0, 0); // b g r
				for (int i = 0; i < n; i++) {
					int srcy = ys + i;
					if (srcy < 0 || srcy >= hs) continue;
					weightY += yw[i];

					Vec3f sumX(0, 0, 0); // b g r 
					float weightX = EPS;
					for (int j = 0; j < n; j++) {
						int srcx = xs + j;
						if (srcx < 0 || srcx >= ws) continue;
						weightX += xw[j];

						Vec3b color = frames[idx].at<Vec3b>(srcy, srcx);
						for (int k = 0; k < 3; k++) sumX[k] += color[k] * xw[j];
					}
					for (int k = 0; k < 3; k++) sumX[k] /= weightX;
					for (int k = 0; k < 3; k++) sum[k] += sumX[k] * yw[i];
				}
				for (int k = 0; k < 3; k++) sum[k] /= weightY;

				for (int k = 0; k < 3; k++) {
					sum[k] = max(min(sum[k], UCHAR_MAX), 0);
					remapFrames[idx].at<Vec3b>(y, x)[k] = sum[k];
				}
			}
		}
	}
}
#undef	CUBIC