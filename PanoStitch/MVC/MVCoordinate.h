#pragma once
#include <vector>
#include <opencv2\opencv.hpp>

namespace MVC_CLONE
{
	struct MvcPerporties
	{
		int BoundarySize;
		int PointSize;
		double *mvcArray;
	};

	struct BCofRegionPoint
	{
		int seq;
		double v1;
		double v2;
		double v3;
		int seq1, seq2, seq3;
	};

	struct MVCTemplate
	{
		int seqNum;
		std::vector<std::vector<BCofRegionPoint> > TriRegionPointSeq;
		std::vector<std::vector<int> > TriangleListSeq;
		std::vector<std::vector<cv::Point>> MVCVertexSeq;
		std::vector<MvcPerporties> MVCoordSeq;
	};

	double CalculateWeights(cv::Point2f p, std::vector<cv::Point> *boundary, double *cudaTriWeightsSeq);

	void findTriangle(std::vector <BCofRegionPoint> &RegionPoint, std::vector<cv::Point> *vertex, std::vector<int> *triList, int width);

	void barycenter(BCofRegionPoint &tmp, int x, int y, cv::Point v1, cv::Point v2, cv::Point v3);

	void CalculateMVCoord(std::vector<cv::Point> *boundary, MvcPerporties *mvc, std::vector<cv::Point> *Coordinates);

	void CalculateMVCoord(std::vector<cv::Point> *boundary, MvcPerporties *mvc, std::vector<cv::Point> *Coordinates, cv::Mat& mask);

	void CalculateMVCoord(std::vector<cv::Point> *boundary, MvcPerporties *mvc, std::vector<cv::Point> *Coordinates, std::vector<int>& mask);

}



