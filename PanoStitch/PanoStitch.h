#pragma once

#include "PanoScript.h"

#include <opencv2\opencv.hpp>
#include <vector>
#include <string>

#define EPS 1e-10

//#define DEBUG_MODE

class PanoStitch {
public:
	PanoStitch(std::string);
	~PanoStitch();

public:
	void setOrder(std::vector<int> inputOrder) { videoOrder = inputOrder; }
	std::vector<int>& getOrder() { return videoOrder; }
	void setDelay(std::vector<int> inputDelay) { videoDelay = inputDelay; }
	std::vector<int>& getDelay() { return videoDelay; }
	void setStartFrameNumber(int number) { startFrameNumber = number; }
	void setEndFrameNumber(int number) { endFrameNumber = number; }
	/*void setMap();
	void getMap();
	void setMask();
	void getMask();*/
	
public:
	static void showImage(std::string, const cv::Mat&, double = 1);
	static void panoFindContours(const cv::Mat&, std::vector<cv::Point>&, int = CV_CHAIN_APPROX_NONE);
public:
	enum position {
		FULL = 0,
		CUT = 1,
		FULL_CEIL = 2
	};
	struct Order {
		position pos;
		int index, min, max;
		bool operator() (Order i, Order j) {
			if (i.pos == FULL_CEIL) return false;
			if (i.min != j.min) return i.min > j.min;
			if (i.max != j.max) return i.max > j.max;
			return true;
		}
	};
public:
	void init(std::string, bool, std::string);
	void stitch(std::string);
protected:
	// precomput remap tables and masks;
	void initMap();
	void initMask(std::string);
	void initOrder();

	void remap(std::vector<cv::Mat>&, std::vector<cv::Mat>&);
	void loadVideo(std::string, std::string);
	void copyInOrder(const std::vector<cv::Mat>&, std::vector<cv::Mat> &, int);
private:
	// private remap functions;
	float cubic01(float x);
	float cubic12(float x);
protected:
	int imNum, w, h;
	PanoScript ps;

	// stitch parameters
	std::vector<int> videoOrder;
	std::vector<int> videoDelay;
	int startFrameNumber, endFrameNumber;
	// preload
	std::vector<cv::Mat> masks;
	std::vector<Order> orders;
	std::vector<cv::VideoCapture> videos;

	cv::Rect dstRoi;
	cv::Mat mapX, mapY;
};