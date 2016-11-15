#pragma once

#include "PanoStitch.h"
#include "MVC\MVCBlender.h"
#include <opencv2\opencv.hpp>

class MVCStitch : public PanoStitch {
public:
	MVCStitch(std::string);
	void initMask(std::string, bool);
	void copyInOrder(const std::vector<cv::Mat>&, std::vector<cv::Mat> &, int);

	void init(std::string, bool, std::string);
	void stitch(std::string);
private:
	// MVC blending
	cv::Mat resultMask;
	
	std::vector<cv::Mat> blendMasks;
	std::vector<std::vector<cv::Point> > boundaries;
	std::vector<std::vector<int> > boundaryIndex;
	MVCBlender *blender;
};