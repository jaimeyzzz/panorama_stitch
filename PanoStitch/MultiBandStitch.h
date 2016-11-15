#pragma once

#include "PanoStitch.h"
#include <opencv2\opencv.hpp>

class MultiBandStitch : public PanoStitch {
public:
	MultiBandStitch(std::string);
	void setMap();
	void setMask(std::string);
	void init(std::string, std::string);
	void stitch(std::string);

private:
	int bandNum;

	std::vector<std::vector<cv::Mat> > weightPyrGaussHost;
	std::vector<cv::Mat> dstPyrLaplaceHost, dstBandWeightsHost;

	cv::Rect dstRoi;
	cv::Point tl, br;
	int top, left, bottom, right;
	cv::Mat resultMask;
};