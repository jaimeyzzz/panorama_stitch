#pragma once

#include "PanoStitch.h"

#include <opencv2\opencv.hpp>
#include <string>

class FeatherStitch : public PanoStitch {
public:
	FeatherStitch(std::string);
	void initMask(std::string, bool);
	void init(std::string, bool, std::string);
	void stitch(std::string);
};
