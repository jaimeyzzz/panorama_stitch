#ifndef IMAGEBLENDING_H
#define IMAGEBLENDING_H

#include "MVC/MVCoordinate.h"
#include <map>
#include <opencv\cv.h>
#include <opencv2\opencv.hpp>
using namespace MVC_CLONE;

class MVCBlender{

public:
	MVCBlender();
	MVCBlender(const cv::Mat& src, const cv::Mat& tgt, const std::vector<cv::Point>& _patchPath);
	MVCBlender(const std::vector<std::vector<cv::Point>>& boundaries, const std::vector<std::vector<int>>& boundaryIndexes);
	~MVCBlender();


	void Init(const cv::Mat& src, const cv::Mat& tgt, const std::vector<cv::Point>& _patchPath);
	

	void ReleaseMemory();

	// ljm add
	void precompute(const std::vector<cv::Mat>&);
	void blend(std::vector<cv::Mat>&);

private:
	void RegionTriangulate(const cv::Mat&);
	void RegionTriangulate(std::vector<int>, const cv::Mat&);
	void CalBoundaryDiff();
	//void CalBoundaryDiffSeq(const std::vector<Mat>&);
	void CalBoundaryDiffSeq(std::vector<cv::Mat>&);

	inline double sqrDist(double d1, double d2)
	{
		return d1 * d1 + d2 * d2;
	}

public:
	cv::Mat blendTrimap;
	cv::Mat blendingResult;
	cv::Mat blendingResultTgt;
	cv::Mat blendingHybridRst;

	cv::Mat sourceResult;
	cv::Mat offsetMapResult;
	cv::Mat offsetMapDouble;

	cv::Mat salientTrimap;

	cv::Mat targetImage;
	cv::Mat sourceImage;

	bool isHybridGrad;
	bool isTriangulate;

	cv::Point pastePos;
	
	std::map<int, std::vector<double> > divHybrid;
	std::map<int, std::vector<cv::Point2f> > gradHybrid;

	int indexN;
	CvMat* matX;	
	std::map<int, int> bandMap;
	std::map<int, int> blendMap;
	int blendN;

	std::vector<std::vector<double> >matRgb;

	MvcPerporties MVCoord;
	std::vector<cv::Point> MVCBoundary;
	std::vector<cv::Point> MVCVertex;
	std::vector<double> boundaryDiff;
	std::vector<double> membrane;

	std::vector<int> TriangleList;
	std::vector<BCofRegionPoint> TriRegionPoint;

	//For MVC stitching
	std::vector<std::vector<cv::Point>> boundarySeq;  // To calculate template
	std::vector<std::vector<int>> boundaryIndexSeq;  // To calculate template

	MVCTemplate stitchingTemplate;
	std::vector<std::vector<double>> boundaryDiffSeq;

	std::vector<MvcPerporties> seamMVCoordSeq;

	std::vector<cv::Point> MVCSeam;
	std::vector<std::vector<cv::Point> > MVCSeamSeq;

	// cuda
	int w, h;
};

#endif // IMAGEBLENDING_H
