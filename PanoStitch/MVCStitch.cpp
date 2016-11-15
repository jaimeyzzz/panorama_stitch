#include "mvcstitch.h"
#include "PanoSeamFinder.h"

#include <ctime>
#include <opencv2\stitching\stitcher.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

// MVC Stitching
MVCStitch::MVCStitch(string scriptName) : PanoStitch(scriptName) {
};

void MVCStitch::initMask(string outdir, bool load) {
	string suffix = "MVC.mask.";
	resultMask = Mat::zeros(h, w, CV_8U);
	if (load) {
		cout << "Set the masks for MVC blending.(load from file)" << endl;
		masks.resize(imNum);
		for (int i = 0; i < imNum; i++) {
			ostringstream ostr;
			ostr << outdir << suffix << i << ".png";
			masks[i] = imread(ostr.str(), 0);
		}
		Mat tmp = Mat::zeros(h, w, CV_8U);
		for (int i = 0; i < imNum; i++) {
			for (int y = 0; y < h; y++) {
				for (int x = 0; x < w; x++) {
					if (masks[i].at<uchar>(y, x) == 255) {
						if (tmp.at<uchar>(y, x) > 0)
							tmp.at<uchar>(y, x) = 128;
						else
							tmp.at<uchar>(y, x) = 255;
						resultMask.at<uchar>(y, x) = 255;
					}
					
				}
			}
		}
		imwrite("tmp\\cover.png", tmp);
		return;
	}
	cout << "Set the masks for MVC blending." << endl;

	PanoSeamFinder pf;
	pf.find(masks, w, h, true);
	// save masks;
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << outdir << suffix << i << ".png";
		imwrite(ostr.str(), masks[i]);
	}
	for (int i = 0; i < imNum; i++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (masks[i].at<uchar>(y, x) == 255) {
					resultMask.at<uchar>(y, x) = 255;
				}
			}
		}
	}
}

void MVCStitch::copyInOrder(const vector<Mat> &src, vector<Mat> &dest, int type) {
	// MVC copyInOder
	for (int i = 0; i < orders.size(); i++) {
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
	}
}

void MVCStitch::init(string inputPath, bool loadmask, string videoformat) {
	initMap();
	initMask(inputPath, loadmask);
	initOrder();
	//loadVideo(inputPath, videoformat);

	copyInOrder(masks, blendMasks, CV_8U);

	boundaries.resize(blendMasks.size() - 1);
	boundaryIndex.resize(blendMasks.size() - 1);

	for (int iter = 1; iter < blendMasks.size(); iter++) {
		vector<Point> contours;
		panoFindContours(blendMasks[iter], contours);
		for (int i = 0; i < contours.size(); i++) {
			int x = contours[i].x, y = contours[i].y;
			if (find(boundaries[iter - 1].begin(), boundaries[iter - 1].end(), Point(x, y)) == boundaries[iter - 1].end())
				boundaries[iter - 1].push_back(Point(x, y));
		}
	}
	Mat tmpMask(h, w, CV_8U);
	if (orders[0].pos == FULL || orders[0].pos == FULL_CEIL) {
		blendMasks[0](Rect(0, 0, w, h)).copyTo(tmpMask);
	}
	else {
		blendMasks[0](Rect(w / 2, 0, w / 2, h)).copyTo(tmpMask(Rect(w / 2, 0, w / 2, h)));
		blendMasks[0](Rect(w, 0, w / 2, h)).copyTo(tmpMask(Rect(0, 0, w / 2, h)));
	}

	const int dx[] = { 0, 1, 1, 1, 0, -1, -1, -1 };
	const int dy[] = { 1, 1, 0, -1, -1, -1, 0, 1 };
	for (int i = 1; i < blendMasks.size(); i++) {
		for (int k = 0; k < boundaries[i - 1].size(); k++) {
			Point pt = boundaries[i - 1][k];
			int ty = pt.y, tx = (pt.x) % w, tx2 = pt.x;
			if (ty < 0 || ty >= h || tx2 < 0 || tx2 >= 2 * w) continue;
			if (blendMasks[i].at<uchar>(ty, tx2) > 0 && (tmpMask.at<uchar>(ty, tx) > 0)) {
				boundaryIndex[i - 1].push_back(k);
			}
		}
		for (int row = 0; row < h; row++) {
			for (int col = 0; col < w * 2; col++) {
				uchar cur = blendMasks[i].at<uchar>(row, col);
				if (cur == 0) continue;
				tmpMask.at<uchar>(row, col % w) = 255;
			}
		}
	}
#if 0 // for debug
	for (int i = 0; i < blendMasks.size() - 1; i++) {
		char str[256];
		sprintf(str, "tmp/mask_%d.png", i + 1);
		imwrite(str, blendMasks[i + 1]);
		Mat tmp = Mat::zeros(h, w, CV_8U);
		for (int j = 0; j < boundaries[i].size(); j++) {
			Point pt = boundaries[i][j];
			tmp.at<uchar>(pt.y, pt.x % w) = 255;
		}
		sprintf(str, "tmp/contours_%d.png", i);
		imwrite(str, tmp);
		tmp = Mat::zeros(h, w, CV_8U);
		for (int j = 0; j < boundaryIndex[i].size(); j++) {
			Point pt = boundaries[i][boundaryIndex[i][j]];
			tmp.at<uchar>(pt.y, pt.x % w) = 255;
		}
		sprintf(str, "tmp/boundary_%d.png", i);
		imwrite(str, tmp);
	}
	//exit(0);
	system("pause");
#endif
	cout << "Precompute MVC oordinates." << endl;
	blender = new MVCBlender(boundaries, boundaryIndex);
	blendMasks.erase(blendMasks.begin());
	blender->precompute(blendMasks);
}

void MVCStitch::stitch(string outputPath) {
	//VideoWriter writer(outfile, CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(w, h));
	bool eof = false;

	cout << "MVC stitching" << endl << "**************************" << endl;
	int count = 0;
	clock_t start = clock();
	vector<Mat> frames(imNum), remapFrames(imNum);
	while (!eof) {
		
		
		/*
		int iw = ps.images[0].w, ih = ps.images[0].h;
		Mat src(ih, iw * imNum, CV_8UC3);
		for (int i = 0; i < imNum; i++) {
			if (!videos[i].read(frames[i])) {
				eof = true;
				break;
			}
			resize(frames[i], frames[i], Size(ps.images[i].w, ps.images[i].h));
			frames[i].copyTo(src(Rect(i * iw, 0, iw, ih)));
		}
		if (eof) break;

		GpuMat cudaSrc;
		cudaSrc.upload(src);
		bend = clock();
		cout << "  load images for : " << (bend - bstart) / cps << " seconds" << endl;
		if (count <= 140) continue;
		if (count > 240) break;
		bstart = clock();
		//remap(frames, remapFrames);*/
		if (count < startFrameNumber) {
			count++;
			continue;
		}
		if (count > endFrameNumber) break;

		for (int i = 0; i < imNum; i++) {
			stringstream istr;
			istr << "Z:\\ljm\\PanoStitch_sample\\scenetest\\images\\remap." << count + videoDelay[videoOrder[i]] << "." << videoOrder[i] << ".png";
			cout << count + videoDelay[videoOrder[i]] << ' ' << videoOrder[i] << endl;
			remapFrames[i] = imread(istr.str());
		}
		clock_t fstart = clock();
		cout << "* stitching for frame " << count << endl;

		blender->blend(remapFrames);

		Mat result = blender->blendingResult;
		result.setTo(Scalar::all(0), resultMask <= 1);
		blender->sourceResult.setTo(Scalar::all(0), resultMask <= 1);
		blender->offsetMapResult.setTo(Scalar::all(0), resultMask <= 1);
		//writer << result;
		cout << "  stitch for : " << (clock() - fstart)  << "ms" << endl;

		stringstream ostr;
		ostr << outputPath << count;
		
		imwrite(ostr.str() + ".png", result);
		//imwrite(ostr.str()+"_blended_mvc.png", result);
		//imwrite(ostr.str()+"_naive_mvc.png", blender->sourceResult);
		//imwrite(ostr.str()+"_offset_mvc.png", abs(blender->offsetMapDouble));
		//imwrite(ostr.str() + "_offset.png", abs(result-blender->sourceResult));
		//imwrite(ostr.str() + "_offsetMap.png", blender->offsetMapResult);

		// offsetMapDouble Mat(CV_64FC3);

		count ++;
	}
	double time = (clock() - start) / 1000.0;
	cout << "*************************************" << endl;
	cout << "stitching " << count - startFrameNumber << " frames in " << time << " s" << endl;
}
