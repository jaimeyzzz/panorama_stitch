#include "MultiBandStitch.h"
#include "PanoSeamFinder.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>

#include <fstream>
#include <iostream>
#include <ctime>
#include <algorithm>

#include "CmFile.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::detail;


// debug : check CUDA erros;
void checkCudaError(cudaError_t err, char * msg);

// MultiBand Stitching
MultiBandStitch::MultiBandStitch(string scriptName) : PanoStitch(scriptName) {
};

void MultiBandStitch::setMap() {
	masks.resize(imNum);
	for (int i = 0; i < imNum; i++) {
		masks[i] = Mat::zeros(ps.pano.h, ps.pano.w, CV_8U);
	}

	std::cout << "Set the frame remapping table." << endl;
	double w2 = (double)ps.pano.w / 2.0 - 0.5, h2 = (double)ps.pano.h / 2.0 - 0.5;

	mapX.create(dstRoi.height, dstRoi.width * imNum, CV_32F);
	mapY.create(dstRoi.height, dstRoi.width * imNum, CV_32F);

	double dx, dy;
	int iw, ih;
	for (int idx = 0; idx < imNum; idx++) {
		ps.setTransPram(idx);
		iw = ps.images[idx].w, ih = ps.images[idx].h;
		double sw2 = (double)ps.images[idx].w / 2.0 - 0.5, sh2 = (double)ps.images[idx].h / 2.0 - 0.5;

		for (int y = 0; y < dstRoi.height; y++) {
			for (int x = 0; x < dstRoi.width; x++) {
				int y_ref = y, x_ref = x;
				if (y >= ps.pano.h) y_ref = 2 * ps.pano.h - 1 - y;
				if (x >= ps.pano.w) x_ref = 2 * ps.pano.w - 1 - x;

				double y_d = (double)y_ref - h2;
				double x_d = (double)x_ref - w2;
				/*double y_d = (double)y - h2;
				double x_d = (double)x - w2;*/
				ps.transform(x_d, y_d, &dx, &dy);
				dx += sw2;
				dy += sh2;
				float fxc = (float)dx, fyc = (float)dy;
				if ((fxc < (float)iw) && (fxc >= 0) && (fyc < (float)ih) && (fyc >= 0)) {
					mapX.at<float>(y, x + idx * dstRoi.width) = fxc;
					mapY.at<float>(y, x + idx * dstRoi.width) = fyc;

					if (y < ps.pano.h && x < ps.pano.w)
						masks[idx].at<uchar>(y, x) = 255;
				}
				else {

					/*mapX.at<float>(y, x + idx * dstRoi.width) = -200;
					mapY.at<float>(y, x + idx * dstRoi.width) = -200;
					continue;*/
					int xs = (int)floor(abs(dx) + 0.5);
					int ys = (int)floor(abs(dy) + 0.5);
					dx = abs(dx) - xs;
					dy = abs(dy) - ys;

					/*if (idx == 0) {
					mapX.at<float>(y, x + idx * dstRoi.width) = rand() % iw;
					mapY.at<float>(y, x + idx * dstRoi.width) = rand() % ih;
					continue;
					}*/
					int xc = xs % (2 * iw), yc = ys % (2 * ih);
					if (xc >= iw) {
						xc = 2 * iw - 1 - xc;
						dx = -dx;
					}
					if (yc >= ih) {
						yc = 2 * ih - 1 - yc;
						dy = -dy;
					}
					mapX.at<float>(y, x + idx * dstRoi.width) = -(dx + (float)xc);
					mapY.at<float>(y, x + idx * dstRoi.width) = -(dy + (float)yc);
				}
			}
		}
	}
	cudaMapX.upload(mapX);
	cudaMapY.upload(mapY);
}

void MultiBandStitch::setMask(string outdir) {
	std::cout << "Set the masks for MultiBand blending." << endl;
	string suffix = "MBB.mask.";

	bool exist = true;
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << outdir << suffix << i << ".png";
		Mat tmp = imread(ostr.str(), 0);
		/*tmp.convertTo(tmp, CV_8U);
		resize(tmp, tmp, Size(w, h));*/
		if (tmp.empty() || (tmp.size() != Size(w, h))) {
			exist = false;
			break;
		}
	}
	if (exist) {
		for (int i = 0; i < imNum; i++) {
			ostringstream ostr;
			ostr << outdir << suffix << i << ".png";
			masks[i] = imread(ostr.str(), 0);
			resize(masks[i], masks[i], Size(w, h));
		}
		//compare(masks[4], 0, masks[4], CV_CMP_GT);
		for (int i = 0; i < imNum; i++) {
			ostringstream ostr;
			ostr << outdir << suffix << i << ".png";
			imwrite(ostr.str(), masks[i]);
		}
		//exit(0);
		return;
	}
	// save masks;
	PanoSeamFinder pf;
	//pf.findVoronoi(masks, w, h);
	pf.find(masks, w, h, false);

	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << outdir << suffix << i << ".png";
		imwrite(ostr.str(), masks[i]);
	}
#define DEBUG_LOG 1
#ifdef DEBUG_LOG
	Mat tmp = Mat::zeros(h, w, CV_8U);
	for (int i = 0; i < imNum; i++) {
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				if (masks[i].at<uchar>(y, x) == 255)
				if (tmp.at<uchar>(y, x) > 0)
					tmp.at<uchar>(y, x) = 128;
				else
					tmp.at<uchar>(y, x) = 255;
			}
		}
	}
	imwrite("tmp\\cover.png", tmp);
#endif
}

void MultiBandStitch::init(string outdir, string videoformat) {
	/*double max_len = static_cast<double>(max(w, h));
	bandNum = static_cast<int>(ceil(log(max_len) / log(2.0)));*/

	int bandW = static_cast<int>(ceil(log((double)w) / log(2.0))), bandH = static_cast<int>(ceil(log((double)h) / log(2.0)));
	bandNum = min(bandW, bandH);

	dstRoi.x = dstRoi.y = 0;
	dstRoi.width = w, dstRoi.height = h;
	dstRoi.width += ((1 << bandNum) - dstRoi.width % (1 << bandNum)) % (1 << bandNum);
	dstRoi.height += ((1 << bandNum) - dstRoi.height % (1 << bandNum)) % (1 << bandNum);
	// precompute mask pyramid;
	srcPyrLaplace.resize(bandNum + 1);
	dstPyrLaplace.resize(bandNum + 1);
	dstBandWeights.resize(bandNum + 1);
	weightPyrGauss.resize(imNum);
	weightPyrMap.resize(bandNum + 1);

	dstPyrLaplace[0].create(dstRoi.size(), CV_16SC3);
	dstPyrLaplace[0].setTo(Scalar::all(255));

	srcPyrLaplace[0].create(dstRoi.height, dstRoi.width * imNum, CV_16SC3);
	srcPyrLaplace[0].setTo(Scalar::all(0));

	dstBandWeights[0].create(dstRoi.size(), CV_32F);
	dstBandWeights[0].setTo(0);

	weightPyrMap[0].create(dstRoi.height, dstRoi.width * imNum, CV_32F);

	for (int i = 1; i <= bandNum; ++i) {
		dstPyrLaplace[i].create((dstPyrLaplace[i - 1].rows + 1) / 2,
			(dstPyrLaplace[i - 1].cols + 1) / 2, CV_16SC3);
		srcPyrLaplace[i].create((srcPyrLaplace[i - 1].rows + 1) / 2,
			(srcPyrLaplace[i - 1].cols + 1) / 2, CV_16SC3);
		weightPyrMap[i].create((weightPyrMap[i - 1].rows + 1) / 2,
			(weightPyrMap[i - 1].cols + 1) / 2, CV_32F);
		dstBandWeights[i].create((dstBandWeights[i - 1].rows + 1) / 2,
			(dstBandWeights[i - 1].cols + 1) / 2, CV_32F);

		dstPyrLaplace[i].setTo(Scalar::all(255));
		dstBandWeights[i].setTo(0);
	}

	int gap = 3 * (1 << bandNum);
	Point tl(0, 0);
	Point tl_new(max(dstRoi.x, tl.x - gap),
		max(dstRoi.y, tl.y - gap));
	Point br_new(min(dstRoi.br().x, tl.x + w + gap),
		min(dstRoi.br().y, tl.y + h + gap));

	tl_new.x = dstRoi.x + (((tl_new.x - dstRoi.x) >> bandNum) << bandNum);
	tl_new.y = dstRoi.y + (((tl_new.y - dstRoi.y) >> bandNum) << bandNum);
	int width = br_new.x - tl_new.x;
	int height = br_new.y - tl_new.y;
	width += ((1 << bandNum) - width % (1 << bandNum)) % (1 << bandNum);
	height += ((1 << bandNum) - height % (1 << bandNum)) % (1 << bandNum);
	br_new.x = tl_new.x + width;
	br_new.y = tl_new.y + height;
	int dy = max(br_new.y - dstRoi.br().y, 0);
	int dx = max(br_new.x - dstRoi.br().x, 0);
	tl_new.x -= dx; br_new.x -= dx;
	tl_new.y -= dy; br_new.y -= dy;

	top = tl.y - tl_new.y;
	left = tl.x - tl_new.x;
	bottom = br_new.y - tl.y - h;
	right = br_new.x - tl.x - w;

	tl.y = tl_new.y - dstRoi.y;
	tl.x = tl_new.x - dstRoi.x;
	br.y = br_new.y - dstRoi.y;
	br.x = br_new.x - dstRoi.x;
	setMap();

	setMask(outdir);
	loadVideo(outdir, videoformat);

	for (int iter = 0; iter < imNum; iter++) {
		GpuMat weightMap, tmpMask;
		tmpMask.upload(masks[iter]);
		tmpMask.convertTo(weightMap, CV_32F, 1. / 255.);

		weightPyrGauss[iter].resize(bandNum + 1);
		copyMakeBorder(weightMap, weightPyrGauss[iter][0], top, bottom, left, right, BORDER_CONSTANT);

		for (int i = 0; i < bandNum; ++i)
			pyrDown(weightPyrGauss[iter][i], weightPyrGauss[iter][i + 1]);

		for (int i = 0; i <= bandNum; ++i) {
			add(weightPyrGauss[iter][i], dstBandWeights[i], dstBandWeights[i]);
		}
	}
	for (int i = 0; i <= bandNum; i++) {
		for (int iter = 0; iter < imNum; iter++) {
			weightPyrGauss[iter][i].copyTo(weightPyrMap[i](Rect(weightPyrGauss[iter][i].cols * iter, 0, weightPyrGauss[iter][i].cols, weightPyrGauss[iter][i].rows)));
		}
	}
	Mat tmp;
	dstBandWeights[0].download(tmp);
	resultMask = tmp > 1e-5f;
	resultMask = resultMask(Range(0, h), Range(0, w));
	std::cout << "findish init" << endl;
}

void MultiBandRemapTex_caller(const GpuMat& src, const GpuMat& dst, GpuMat& mapX, GpuMat& mapY, int N, int ws, int wd, cudaStream_t stream);
void MultiBandPyrCreate_caller(const GpuMat& src, const GpuMat& dst, int w, int h, int N, cudaStream_t stream);
void MultiBandPyrBlend_caller(GpuMat& src, GpuMat& dst, GpuMat& weight, GpuMat& bandWeight, int imNum, cudaStream_t stream);
void MultiBandPyrRestore_caller(GpuMat& src, GpuMat& dst, cudaStream_t stream);

void MultiBandStitch::stitch(string outdir) {
	if (~CmFile::FolderExist(outdir)) {
		CmFile::MkDir(outdir);
	}
	VideoWriter writer((outdir + "MBB.avi").c_str(), CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(w, h));
	bool eof = false;

	std::cout << "MultiBand stitching" << endl << "**************************" << endl;
	int count = 0;
	clock_t start = clock();

	vector<GpuMat> tmpPyr(bandNum + 1);
	for (int i = 0; i <= bandNum; i++)
		tmpPyr[i].create(dstPyrLaplace[i].size(), dstPyrLaplace[i].type());

	//// 调整帧同步
	/*Mat tmpFrame;
	for (int j = 0; j < 6; j++) {
	for (int i = 0; i < 0; i++) {
	if (j != 2 || j != 4) {
	videos[j].read(tmpFrame);
	}
	}
	}*/
	/*VideoWriter writers[6];
	for (int i = 0; i < 6; i++) {
	stringstream str;
	str << outdir << "origin_" << i << ".avi";
	writers[i].open(str.str().c_str(), CV_FOURCC('M', 'J', 'P', 'G'), 30.0, Size(ps.images[i].w, ps.images[i].h));;
	}*/
	while (!eof) {
		std::cout << "* stitching for frame " << count << endl;
		clock_t ostart = clock(), bstart = clock(), bend;
		bend = clock();

		vector<Mat> frames(imNum);
		int iw = ps.images[0].w, ih = ps.images[0].h;
		Mat src(ih, iw * imNum, CV_8UC3), src_s;
		for (int i = 0; i < imNum; i++) {
			if (!videos[i].read(frames[i])) {
				cout << "stop while reading video : " << i << endl;
				eof = true;
				break;
			}
			resize(frames[i], frames[i], Size(ps.images[i].w, ps.images[i].h));

			frames[i].copyTo(src(Rect(i * iw, 0, iw, ih)));
		}
		if (eof) break;

		count++;
		//if (count < 140) continue;
		if (count >= 540) break;

		/*for (int i = 0; i < imNum; i ++)
		writers[i] << frames[i];
		*/
		src.convertTo(src_s, CV_16S, 1 << 7);
		GpuMat cudaSrc;
		ostart = clock();
		cudaSrc.upload(src_s);
		cudaDeviceSynchronize();
		std::cout << "  upload for : " << clock() - ostart << endl;
		bstart = clock();
		// multiband blending process begin;

		float time_elapsed;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		MultiBandRemapTex_caller(cudaSrc, srcPyrLaplace[0], cudaMapX, cudaMapY, imNum, iw, dstRoi.width, StreamAccessor::getStream(Stream::Null()));

		Mat remapSrc(srcPyrLaplace[0]);
		for (int i = 0; i < imNum; i++) {
			remapSrc(Rect(i * dstRoi.width, 0, w, h)).copyTo(frames[i]);
		}
		for (int i = 0; i < imNum; i++) {
			Mat tmp = Mat::zeros(h, w, CV_16SC3);
			frames[i].copyTo(tmp);
			tmp.convertTo(tmp, CV_8U, 1. / 128.);
			ostringstream out;
			out << outdir << "remap." << count << "." << i << ".png";
			imwrite(out.str().c_str(), tmp);
		}
		for (int i = 0; i < bandNum; i++) {
			MultiBandPyrCreate_caller(srcPyrLaplace[i], srcPyrLaplace[i + 1], dstPyrLaplace[i].cols, dstPyrLaplace[i].rows, imNum, StreamAccessor::getStream(Stream::Null()));

		}
		for (int i = 0; i <= bandNum; i++) {
			MultiBandPyrBlend_caller(srcPyrLaplace[i], dstPyrLaplace[i], weightPyrMap[i], dstBandWeights[i], imNum, StreamAccessor::getStream(Stream::Null()));

		}
		for (size_t i = dstPyrLaplace.size() - 1; i > 0; --i) {
			MultiBandPyrRestore_caller(dstPyrLaplace[i], dstPyrLaplace[i - 1], StreamAccessor::getStream(Stream::Null()));
		}
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(start);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time_elapsed, start, stop);
		cout << "  stitch for  : " << time_elapsed << endl;
		// multiband blending process end;
		bstart = clock();
		Mat result(dstPyrLaplace[0]);
		cudaDeviceSynchronize();
		bend = clock();
		std::cout << "  download for : " << (bend - bstart) / (double)CLOCKS_PER_SEC << " seconds" << endl;
		//dstPyrLaplace[0].download(result);

		result = result(Range(0, h), Range(0, w));
		//result.setTo(Scalar::all(0), resultMask <= 1);
		result.convertTo(result, CV_8UC3, 1. / 128.);

		writer << result;

		ostringstream out;
		out << outdir << count << ".jpg";
		imwrite(out.str().c_str(), result);

		//showImage(result);
	}
	double ftime = (clock() - start) / cps;
	std::cout << "*************************************" << endl;
	std::cout << "stitching " << count << " frames in " << ftime << " seconds" << endl;
}