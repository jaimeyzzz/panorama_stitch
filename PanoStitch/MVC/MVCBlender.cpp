#include "MVCTriangulate.h"
#include "MVCBlender.h"
#include "PanoStitch.h"

#include <fstream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

MVCBlender::MVCBlender()
{
	MVCoord.mvcArray = NULL;
	matX = NULL;

	isHybridGrad = false;
	isTriangulate = false;
	pastePos = Point(0, 0);
}

MVCBlender::MVCBlender(const vector<vector<Point>>& _boundarySeq, const vector<vector<int>>& _boundaryIndexSeq)
	:boundarySeq(_boundarySeq),boundaryIndexSeq(_boundaryIndexSeq)
{
	
}
// this is invalid anymore
MVCBlender::MVCBlender(const Mat& _src, const Mat& _tgt, const vector<Point>& _patchPath)
{
	sourceImage = _src;
	targetImage = _tgt;

	blendTrimap = Mat(_src);
	blendTrimap.zeros(blendTrimap.size(), blendTrimap.type());	
	
	MVCBoundary = _patchPath;
	MVCVertex = _patchPath;
	MVCoord.mvcArray = NULL;
	
	blendingResult = _src;
	blendingResultTgt = _tgt;
	pastePos = Point(0, 0);

	matX = NULL;

	isHybridGrad = false;
	isTriangulate = false;
}

void MVCBlender::Init(const Mat& _src, const Mat& _tgt, const vector<Point>& _patchPath)
{
	sourceImage = _src;
	targetImage = _tgt;

	blendTrimap = Mat(_src);
	blendTrimap.zeros(blendTrimap.size(), blendTrimap.type());

	salientTrimap = _src;
	salientTrimap.zeros(salientTrimap.size(), salientTrimap.type());

	MVCBoundary = _patchPath;
	MVCVertex = _patchPath;

	blendingResult = _src;
	blendingResultTgt = _tgt;
	pastePos = Point(0, 0);	

	isHybridGrad = false;
	isTriangulate = false;

	divHybrid.clear();
	gradHybrid.clear();

	if (matX) cvReleaseMat(&matX);
	if (MVCoord.mvcArray) {
		delete MVCoord.mvcArray;
		MVCoord.mvcArray = NULL;
	}

}

MVCBlender::~MVCBlender()
{
	if (matX) cvReleaseMat(&matX);
	if (MVCoord.mvcArray) {
		delete MVCoord.mvcArray;
		MVCoord.mvcArray = NULL;
	}
	MVCBoundary.clear();
	MVCVertex.clear();
	TriRegionPoint.clear();
	divHybrid.clear();
	gradHybrid.clear();
}

void MVCBlender::ReleaseMemory()
{
	if (matX) cvReleaseMat(&matX);
	if (MVCoord.mvcArray) 
	{
		delete MVCoord.mvcArray;
		MVCoord.mvcArray = NULL;
	}

	MVCBoundary.clear();
	MVCVertex.clear();
	TriRegionPoint.clear();
	divHybrid.clear();
	gradHybrid.clear();
}


//////////////////////////////////////////////////////////////////////////
// MVC Stitching
void MVCBlender::precompute(const vector<Mat>& masks)
{
	w = masks[0].cols / 2, h = masks[0].rows;

	stitchingTemplate.MVCVertexSeq.clear();
	stitchingTemplate.TriangleListSeq.clear();
	stitchingTemplate.TriRegionPointSeq.clear();
	stitchingTemplate.MVCoordSeq.clear();

	int stitchingNum = boundarySeq.size();
	stitchingTemplate.seqNum = stitchingNum;

	for (int iter = 0; iter < stitchingNum; iter++) {
		vector<Point> boundary = boundarySeq[iter];
		vector<int> boundaryIndex = boundaryIndexSeq[iter];

		MVCBoundary = boundary;
		MVCVertex = boundary;
		MVCoord.mvcArray = NULL;

		RegionTriangulate(boundaryIndex, masks[iter]);

		stitchingTemplate.MVCVertexSeq.push_back(MVCVertex);
		stitchingTemplate.TriangleListSeq.push_back(TriangleList);
		stitchingTemplate.TriRegionPointSeq.push_back(TriRegionPoint);
		stitchingTemplate.MVCoordSeq.push_back(MVCoord);
	}
	for (int iter = 0; iter < stitchingNum - 1; iter++) {
		vector<Point> boundary = boundarySeq[iter];
		vector<int> boundaryIndex = boundaryIndexSeq[iter];

		MVCBoundary = boundarySeq[iter + 1]; // Seam is included here;
		MVCSeam.clear();
		for (int i = iter + 1; i < boundarySeq.size(); i++) {
			for (int j = 0; j < boundarySeq[i].size(); j++) {
				Point& pt = boundarySeq[i].at(j);
				if (masks[iter].at<uchar>(pt.y, pt.x) != 0) {
					MVCSeam.push_back(pt);
				}
				if (masks[iter].at<uchar>(pt.y, (pt.x + w) % (2 * w)) != 0) {
					MVCSeam.push_back(Point((pt.x + w) % (2 * w), pt.y));
				}
			}
		}
		MVCSeamSeq.push_back(MVCSeam);
		MVCoord.PointSize = MVCSeam.size();
		MVCoord.BoundarySize = boundary.size();

		MVCoord.mvcArray = NULL;
		MVCoord.mvcArray = new double[MVCoord.PointSize * MVCoord.BoundarySize];

		CalculateMVCoord(&boundary, &MVCoord, &MVCSeam, boundaryIndex);
		seamMVCoordSeq.push_back(MVCoord);
	}
}

void MVCBlender::blend(vector<Mat>& imgSeq)
{
	//Calculate seams first
	CalBoundaryDiffSeq(imgSeq);

	blendingResult = imgSeq[0].clone();
	sourceResult = imgSeq[0].clone();
	offsetMapResult = Mat::zeros(sourceResult.size(), sourceResult.type()) + Scalar(127, 127, 127);

	offsetMapDouble = Mat::zeros(offsetMapResult.size(), CV_64FC3);
	for (int iter = 0; iter < stitchingTemplate.seqNum; iter++) {
		MVCoord = stitchingTemplate.MVCoordSeq[iter];
		TriRegionPoint = stitchingTemplate.TriRegionPointSeq[iter];
		TriangleList = stitchingTemplate.TriangleListSeq[iter];
		MVCVertex = stitchingTemplate.MVCVertexSeq[iter];

		targetImage = blendingResult;
		sourceImage = imgSeq[iter + 1];

		membrane.clear();
		int RegionSize = MVCoord.PointSize;
		int BoundarySize = MVCoord.BoundarySize;

		MVCBoundary = boundarySeq[iter];

		boundaryDiff = boundaryDiffSeq[iter];
		//CalBoundaryDiff();

		for (int i = 0; i < RegionSize; ++i)
		{
			double r = 0;
			double g = 0;
			double b = 0;

			double *temp = MVCoord.mvcArray + i * BoundarySize;
			for (int k = 0; k < BoundarySize; ++k)
			{
				double t = *(temp + k);
				r += t * boundaryDiff.at(3 * k);
				g += t * boundaryDiff.at(3 * k + 1);
				b += t * boundaryDiff.at(3 * k + 2);
			}
			membrane.push_back(r);
			membrane.push_back(g);
			membrane.push_back(b);
		}

		BCofRegionPoint pixel;
		Vec3b resultColor;

		for (int i = 0; i < TriRegionPoint.size(); ++i)
		{
			pixel = TriRegionPoint[i];

			int seq1 = TriangleList[pixel.seq * 3];
			int seq2 = TriangleList[pixel.seq * 3 + 1];
			int seq3 = TriangleList[pixel.seq * 3 + 2];

			int x = pixel.v1 * MVCVertex[seq1].x +
				pixel.v2 * MVCVertex[seq2].x +
				pixel.v3 * MVCVertex[seq3].x + 0.5;

			int y = pixel.v1 * MVCVertex[seq1].y +
				pixel.v2 * MVCVertex[seq2].y +
				pixel.v3 * MVCVertex[seq3].y + 0.5;

			x = x % w;
			if (!Rect(Point(0, 0), targetImage.size()).contains(Point(x, y)))
				continue;

			double r = pixel.v1 * membrane[seq1 * 3] + pixel.v2 * membrane[seq2 * 3] +
				pixel.v3 * membrane[seq3 * 3];

			double g = pixel.v1 * membrane[seq1 * 3 + 1] + pixel.v2 * membrane[seq2 * 3 + 1] +
				pixel.v3 * membrane[seq3 * 3 + 1];

			double b = pixel.v1 * membrane[seq1 * 3 + 2] + pixel.v2 * membrane[seq2 * 3 + 2] +
				pixel.v3 * membrane[seq3 * 3 + 2];

			resultColor = sourceImage.at<Vec3b>(y, x);
			// set origin source image;
			sourceResult.at<Vec3b>(y, x) = sourceImage.at<Vec3b>(y, x);

			// set offsetmap
			offsetMapResult.at<Vec3b>(y, x) = Vec3b(MAX(MIN((127 + b), 255), 0), MAX(MIN((127 + g), 255), 0), MAX(MIN((127 + r), 255), 0));
			
			offsetMapDouble.at<Vec3d>(y, x) = Vec3d(b, g, r);

			double rr = MAX(MIN((resultColor[2] + r), 255), 0);
			double rg = MAX(MIN((resultColor[1] + g), 255), 0);
			double rb = MAX(MIN((resultColor[0] + b), 255), 0);

			resultColor[2] = (uchar)rr; //.setRedF(MAX(MIN((resultColor.redF() + r), 1), 0));
			resultColor[1] = (uchar)rg; //.setGreenF(MAX(MIN((resultColor.greenF() + g), 1), 0));
			resultColor[0] = (uchar)rb; // .setBlueF(MAX(MIN((resultColor.blueF() + b), 1), 0));

			blendingResult.at<Vec3b>(y, x) = resultColor; //.setPixel(x, y, resultColor.rgb());
		}
		/*ofstream fout("debug\\mvc_offset_map.txt");
		for (int i = 0; i < offsetMapDouble.rows; i ++) {
			for (int j = 0; j < offsetMapDouble.cols; j ++) {
				for (int k = 0; k < 3; k++) {
					fout << offsetMapDouble.at<Vec3d>(i, j)[k] << ' ';
				}
			}
			fout << endl;
		}
		fout.close();*/
#ifdef DEBUG_MODE
		imwrite("debug\\mvc_blending_result.png", blendingResult);
		PanoStitch::showImage("blendingResult", blendingResult, 0.8);
#endif
	}
}

void MVCBlender::CalBoundaryDiff()
{
	boundaryDiff.clear();

	for (int i = 0; i < MVCBoundary.size(); ++i)
	{
		Point pt_s = MVCBoundary.at(i);
		Point pt_t = pt_s + pastePos;

		if (!Rect(Point(0, 0), targetImage.size()).contains(pt_t))
		{
			for (int j = 0; j < 3; ++j)
				boundaryDiff.push_back(0.0);
			continue;
		}

		Vec3b color_s = sourceImage.at<Vec3b>(pt_s.y, pt_s.x);
		Vec3b color_t = targetImage.at<Vec3b>(pt_t.y, pt_t.x);

		boundaryDiff.push_back(color_t[2] - color_s[2]);
		boundaryDiff.push_back(color_t[1] - color_s[1]);
		boundaryDiff.push_back(color_t[0] - color_s[0]);
	}
}


void MVCBlender::CalBoundaryDiffSeq(vector<Mat>& imgSeq) {
	boundaryDiffSeq.clear();
	Mat boundaryImg;

	imgSeq[0].copyTo(boundaryImg);
	for (int iter = 0; iter < stitchingTemplate.seqNum; iter++) {
		MVCBoundary = boundarySeq[iter];
		boundaryDiff.resize(MVCBoundary.size() * 3);
		for (int i = 0; i < boundaryDiff.size(); i++) {
			boundaryDiff[i] = 0;
		}
		vector<int> boundaryIndex = boundaryIndexSeq[iter];
		for (int i = 0; i < MVCBoundary.size(); i++) {
		//for (int i = 0; i < boundaryIndex.size(); i++) {
			//int idx = boundaryIndex[i];
			int idx = i;
			Point pt = MVCBoundary[idx];

			Vec3b color_s = imgSeq[iter + 1].at<Vec3b>(pt.y, pt.x % w);
			Vec3b color_t = boundaryImg.at<Vec3b>(pt.y, pt.x % w);

			boundaryDiff[idx * 3] = color_t[2] - color_s[2];
			boundaryDiff[idx * 3 + 1] = color_t[1] - color_s[1];
			boundaryDiff[idx * 3 + 2] = color_t[0] - color_s[0];
		}
		boundaryDiffSeq.push_back(boundaryDiff);
		if (iter + 1 == stitchingTemplate.seqNum) break;

		targetImage = boundaryImg;
		sourceImage = imgSeq[iter + 1];

		MVCoord = seamMVCoordSeq[iter];

		int RegionSize = MVCoord.PointSize;
		int BoundarySize = MVCoord.BoundarySize;

		MVCBoundary = boundarySeq[iter];
		MVCSeam = MVCSeamSeq[iter];
		Vec3b resultColor;

		for (int i = 0; i < RegionSize; ++i)
		{
			double r = 0;
			double g = 0;
			double b = 0;

			double *temp = MVCoord.mvcArray + i * BoundarySize;
			for (int k = 0; k < BoundarySize; ++k)
			{
				double t = *(temp + k);
				r += t * boundaryDiff.at(3 * k);
				g += t * boundaryDiff.at(3 * k + 1);
				b += t * boundaryDiff.at(3 * k + 2);
			}
			Point & pt = MVCSeam.at(i);
			resultColor = sourceImage.at<Vec3b>(pt.y, pt.x % w);
			double rr = MAX(MIN((resultColor[2] + r), 255), 0);
			double rg = MAX(MIN((resultColor[1] + g), 255), 0);
			double rb = MAX(MIN((resultColor[0] + b), 255), 0);

			resultColor[2] = (uchar)rr; 
			resultColor[1] = (uchar)rg; 
			resultColor[0] = (uchar)rb; 

			boundaryImg.at<Vec3b>(pt.y, pt.x % w) = resultColor;
		}
	}
}

void MVCBlender::RegionTriangulate(const Mat &mask)
{
	TriangleList.clear();
	
	MVCTriangulate(MVCVertex, TriangleList, mask);

	MVCoord.PointSize = MVCVertex.size();
	MVCoord.BoundarySize = MVCBoundary.size();

	if (MVCoord.mvcArray) delete MVCoord.mvcArray;	
	MVCoord.mvcArray = new double[MVCoord.PointSize * MVCoord.BoundarySize];

	vector<int> boundaryIndex;
	
	for (int i = 0; i < MVCBoundary.size(); ++i) {
		Point pt = MVCBoundary[i];

		if (Rect(Point(0, 0), targetImage.size()).contains(pt + pastePos)) {
			boundaryIndex.push_back(i);
		}
	}

	CalculateMVCoord(&MVCBoundary, &MVCoord, &MVCVertex, boundaryIndex);
	TriRegionPoint.clear();
	findTriangle(TriRegionPoint, &MVCVertex, &TriangleList, sourceImage.cols);
}

void MVCBlender::RegionTriangulate(vector<int> boundaryIndex, const Mat &mask) {
	TriangleList.clear();

	MVCTriangulate(MVCVertex, TriangleList, mask);

	MVCoord.PointSize = MVCVertex.size();
	MVCoord.BoundarySize = MVCBoundary.size();

	if (MVCoord.mvcArray) delete MVCoord.mvcArray;	
	MVCoord.mvcArray = new double[MVCoord.PointSize * MVCoord.BoundarySize];
	memset(MVCoord.mvcArray, 0.0, sizeof(double) * MVCoord.PointSize * MVCoord.BoundarySize);

	CalculateMVCoord(&MVCBoundary, &MVCoord, &MVCVertex, boundaryIndex);
	TriRegionPoint.clear();
	findTriangle(TriRegionPoint, &MVCVertex, &TriangleList, 0);
}