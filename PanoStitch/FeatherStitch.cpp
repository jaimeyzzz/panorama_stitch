#include "featherstitch.h"
#include <ctime>

#include <opencv2\stitching\stitcher.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

// Feather Stitching
FeatherStitch::FeatherStitch(string scriptName) : PanoStitch(scriptName) {
};

void FeatherStitch::initMask(string outdir, bool tryLoad) {
	cout << "Set the masks for Feather blending." << endl;
	string suffix = "Feather.Mask.";
	bool exist = tryLoad;
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << outdir << suffix << i << ".png";
		Mat tmp = imread(ostr.str(), 0);
		if (tmp.empty() || tmp.size() != Size(w, h)) {
			exist = false;
			break;
		}
	}
	if (exist) {
		for (int i = 0; i < imNum; i++) {
			ostringstream ostr;
			ostr << outdir << suffix << i << ".png";
			masks[i] = imread(ostr.str(), 0);
		}
		return;
	}
	// save masks;
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << outdir << suffix << i << ".png";
		imwrite(ostr.str(), masks[i]);
	}
#ifdef DEBUG_MODE
	Mat maskCover = Mat::zeros(h, w, CV_8U);
	for (int i = 0; i < imNum; i++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (masks[i].at<uchar>(y, x) == 255)
				if (maskCover.at<uchar>(y, x) > 0)
					maskCover.at<uchar>(y, x) = 128;
				else
					maskCover.at<uchar>(y, x) = 255;
			}
		}
	}
	imwrite("debug\\feather_mask_cover.png", maskCover);
#endif
}

void FeatherStitch::init(string outdir, bool tryLoad, string videoformat) {
	initMap();
	initMask(outdir, tryLoad);
	//loadVideo(outdir, videoformat);
}

void FeatherStitch::stitch(string outputPath) {
	//VideoWriter writer(outputPath + "feather.avi", CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(w, h));
	bool eof = false;

	cout << "Feather stitching" << endl << "**************************" << endl;
	int count = 140;
	clock_t start = clock();

	while (!eof) {
		cout << "* stitching for frame " << count << endl;
		if (count > endFrameNumber) {
			break;
		}

		vector<Mat>frames(imNum), remapFrames(imNum);

		int loadRemapFrames = 1;
		if (loadRemapFrames) {
			// load remap frames;
			for (int i = 0; i < imNum; i++) {
				stringstream istr;
				istr << "Z:\\ljm\\PanoStitch_sample\\scenetest\\images\\remap." << count + videoDelay[i] << "." << i << ".png";
				cout << istr.str() << endl;
				remapFrames[i] = imread(istr.str());
				/*imshow("", remapFrames[i]);
				waitKey();*/
			}
		}
		else {
			int iw = ps.images[0].w, ih = ps.images[0].h;
			Mat src(ih, iw * imNum, CV_8UC3);
			for (int i = 0; i < imNum; i++) {
				if (!videos[i].read(frames[i])) {
					eof = true;
					break;
				}
				resize(frames[i], frames[i], Size(ps.images[i].w, ps.images[i].h));
			}
			if (eof) break;
			remap(frames, remapFrames);
		}
		clock_t ostart = clock(), bend;
		
		//if (count < 140) continue;
		//if (count >= 440) break;
		double shiftOffset = 1.0;
		for (int i = 0; i < imNum; i++)
			remapFrames[i].convertTo(remapFrames[i], CV_16SC3, shiftOffset);

		MultiBandBlender blender(false, 100);
		//FeatherBlender blender(0.01);
		blender.prepare(Rect(0, 0, w, h));

		for (int i = 0; i < imNum; i++) {
			blender.feed(remapFrames[i], masks[i], Point(0, 0));
		}
		Mat result, resultMask;
		blender.blend(result, resultMask);
		result.convertTo(result, CV_8UC3, 1 / shiftOffset);

		//writer << result;
		cout << "  stitch for : " << (clock() - ostart)  << " s" << endl;

		stringstream ostr;
		ostr << outputPath << count << ".png";
		imwrite(ostr.str(), result);

		count++;
	}
	cout << "*************************************" << endl;
	cout << "stitching " << count << " frames in " << (clock() - start) / 1000.0 << " s" << endl;
}
