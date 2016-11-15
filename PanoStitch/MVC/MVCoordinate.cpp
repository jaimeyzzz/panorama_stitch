#include "MVCoordinate.h"
#include "Vector3f.h"
#include <iostream>
using namespace std;
using namespace cv;

namespace MVC_CLONE
{
	const double EPSILON = 1e-6;

	double CalculateWeights(Point2f p, vector<Point> *boundary, double *weightSeq)
	{
		double sum = 0.;
		int size = boundary->size() - 1;
		CVector3f pointX;
		CVector3f p0, p1, p2;
		CVector3f v1, v2;

		pointX.x = p.x + EPSILON;
		pointX.y = p.y + EPSILON;
		pointX.z = .0;

		memset(weightSeq, 0, sizeof(double) * size);

		for (int i = 0; i < size; i++)
		{
			p0.x = boundary->at((i - 1 + size) % size).x;
			p0.y = boundary->at((i - 1 + size) % size).y;
			p0.z = .0;

			p1.x = boundary->at(i).x;
			p1.y = boundary->at(i).y;
			p1.z = 0.00001;

			p2.x = boundary->at((i + 1 + size) % size).x;
			p2.y = boundary->at((i + 1 + size) % size).y;
			p2.z = .0;

			v1 = p0 - pointX;
			v2 = p1 - pointX;

			double angle1 = AngleBetweenVectors(v1, v2);
			if (v1.x * v2.y - v1.y * v2.x > 0)
				angle1 = -angle1;

			v1 = v2;
			v2 = p2 - pointX;

			double angle2 = AngleBetweenVectors(v1, v2);
			if (v1.x * v2.y - v1.y * v2.x > 0)
				angle2 = -angle2;

			weightSeq[i] = (tan(angle1 / 2.) + tan(angle2 / 2.)) / (Distance(p1, pointX) + 0.001);
			sum += weightSeq[i];
		}

		return sum;
	}

	void findTriangle(vector <BCofRegionPoint> &RegionPoint, vector<Point> *vertex, vector<int> *triList, int width)
	{
		int left, right, up, bottom;
		Point vertex1,vertex2,vertex3;
		BCofRegionPoint tmp;

		// QFile file("Vertex.txt");
		// if (!file.open(QIODevice::ReadWrite | QIODevice::Text))
		//	 return;
		// QTextStream out(&file);	

		for(int i = 0; i < triList->size() / 3; ++i)
		{
			//QPainterPath triangle;
			vertex1 = vertex->at(triList->at(i * 3));
			vertex2 = vertex->at(triList->at(i * 3 + 1));
			vertex3 = vertex->at(triList->at(i * 3 + 2));

			//out << i <<":"<<triList->at(i*3)<<" "<<triList->at(i*3 + 1)<<" "<<triList->at(i*3 + 2)<<"\n";
			//file.flush();

			/*triangle.moveTo(vertex1);
			triangle.lineTo(vertex2);
			triangle.lineTo(vertex3);*/
			//triangle.lineTo(vertex1);
			
			tmp.seq = i;
			tmp.v1 = 1.;
			tmp.v2 = 0.;
			tmp.v3 = 0.;
			RegionPoint.push_back(tmp);
			
			tmp.v1 = 0.;
			tmp.v2 = 1.;
			tmp.v3 = 0.;
			RegionPoint.push_back(tmp);
			
			tmp.v1 = 0.;
			tmp.v2 = 0.;
			tmp.v3 = 1.;
			RegionPoint.push_back(tmp);

			left = MIN(MIN(vertex1.x,vertex2.x),vertex3.x);
			right = MAX(MAX(vertex1.x,vertex2.x),vertex3.x);
			up = MIN(MIN(vertex1.y,vertex2.y),vertex3.y);
			bottom = MAX(MAX(vertex1.y,vertex2.y),vertex3.y);

			vector<Point> contours;
			contours.push_back(vertex1);
			contours.push_back(vertex2);
			contours.push_back(vertex3);
			contours.push_back(vertex1);
			/*cv::Mat rp(bottom-up+1,right-left+1,CV_8UC3);
			rp.setTo(0);
			std::ofstream of("qt.txt");
			int count = 0;*/
			for(int n = up; n <= bottom; ++n)
			{
				for(int m = left; m <= right; ++m)
				{
					//if(triangle.contains(Point(m,n)))
					if (pointPolygonTest(contours, Point(m, n), false) >= 0)
					{
						tmp.seq = i;
						barycenter(tmp, m, n, vertex1,vertex2,vertex3);
						RegionPoint.push_back(tmp);
						/*rp.at<cv::Vec3b>(n-up,m-left) = cv::Vec3b(255,255,255);
						of<<m<<" "<<n<<";";
						count++;*/
					}
				}
			}
			/*of<<std::endl<<count;
			of.close();
			cv::imwrite("rp.png",rp);*/
		}

		//	file.close();

	}

	void barycenter(BCofRegionPoint &tmp, int x, int y, Point v1, Point v2, Point v3)
	{
		CVector3f a1(v1.x-v2.x,v1.y-v2.y,0);	//v1-v2
		CVector3f a2(v3.x-v2.x,v3.y-v2.y,0);	//v3-v2

		double s = Magnitude(Cross(a1,a2));

		a1.x = v1.x-x;
		a1.y = v1.y-y;	
		a2.x = v2.x-x;
		a2.y = v2.y-y;

		double s1 = Magnitude(Cross(a1,a2));
		tmp.v3 = s1 / s;

		a2.x = v3.x-x;
		a2.y = v3.y-y;

		s1 = Magnitude(Cross(a1,a2));
		tmp.v2 = s1 / s;

		a1.x = v2.x-x;
		a1.y = v2.y-y;


		s1 = Magnitude(Cross(a1,a2));
		tmp.v1 = s1 / s;

	/*
		SuperMatrix A;
		double rhs[3] = {1.0,2.0,3.0};
		dCreate_Dense_Matrix(&A, 3, 1, rhs, 1, SLU_DN, SLU_D, SLU_GE);

		int a =0;

	*/
	}
	
	void CalculateMVCoord(vector<Point> *boundary, MvcPerporties *mvc, vector<Point> *Coordinates)
	{
		int BoundarySize = mvc->BoundarySize;
		double *mvcArray = mvc->mvcArray;

		double *weightArray = new double[BoundarySize];

		int vertexSize = Coordinates->size();

		for (int i = 0; i < vertexSize; ++i)
		{
			double sum = CalculateWeights(Point2f(Coordinates->at(i)), boundary, weightArray);
			for (int j = 0; j < BoundarySize; ++j)
				*(mvcArray + i * BoundarySize + j) = weightArray[j] / sum;
		}

		delete[] weightArray;
	}



	//////////////////////////////////////////////////////////////////////////

	double CalculateWeightsMask(Point2f p, vector<Point> *boundary, double *weightSeq, Mat& mask)
	{
		double sum = 0.;
		int size = boundary->size() - 1;
		CVector3f pointX;
		CVector3f p0, p1, p2;
		CVector3f v1, v2;

		pointX.x = p.x + EPSILON;
		pointX.y = p.y + EPSILON;
		pointX.z = .0;

		memset(weightSeq, 0, sizeof(double) * size);

		for (int i = 0; i < size; i++)
		{
			Point pt = boundary->at(i);

			if (mask.at<uchar>(pt.y, pt.x) < 200) // if the boundary point is not marked
			{
				weightSeq[i] = 0;
				continue;
			}			

			p0.x = boundary->at((i - 1 + size) % size).x;
			p0.y = boundary->at((i - 1 + size) % size).y;
			p0.z = .0;

			p1.x = boundary->at(i).x;
			p1.y = boundary->at(i).y;
			p1.z = 0.00001;

			p2.x = boundary->at((i + 1 + size) % size).x;
			p2.y = boundary->at((i + 1 + size) % size).y;
			p2.z = .0;

			v1 = p0 - pointX;
			v2 = p1 - pointX;

			double angle1 = AngleBetweenVectors(v1, v2);
			if (v1.x * v2.y - v1.y * v2.x > 0)
				angle1 = -angle1;

			v1 = v2;
			v2 = p2 - pointX;

			double angle2 = AngleBetweenVectors(v1, v2);
			if (v1.x * v2.y - v1.y * v2.x > 0)
				angle2 = -angle2;

			weightSeq[i] = (tan(angle1 / 2.) + tan(angle2 / 2.)) / (Distance(p1, pointX) + 0.001);
			sum += weightSeq[i];
		}

		return sum;
	}

	double CalculateWeightsVec(Point2f p, vector<Point> *boundary, double *weightSeq, vector<int>& mask)
	{
		double sum = 0.;
		int size = boundary->size() - 1;
		CVector3f pointX;
		CVector3f p0, p1, p2;
		CVector3f v1, v2;

		pointX.x = p.x + EPSILON;
		pointX.y = p.y + EPSILON;
		pointX.z = .0;

		memset(weightSeq, 0, sizeof(double) * size);

		for (int n = 0; n < mask.size(); ++n)
		{
			int i = mask[n];

			Point pt = boundary->at(i);				

			p0.x = boundary->at((i - 1 + size) % size).x;
			p0.y = boundary->at((i - 1 + size) % size).y;
			p0.z = .0;

			p1.x = boundary->at(i).x;
			p1.y = boundary->at(i).y;
			p1.z = 0.00001;

			p2.x = boundary->at((i + 1 + size) % size).x;
			p2.y = boundary->at((i + 1 + size) % size).y;
			p2.z = .0;

			v1 = p0 - pointX;
			v2 = p1 - pointX;

			double angle1 = AngleBetweenVectors(v1, v2);
			/*if (v1.x * v2.y - v1.y * v2.x > 0)
				angle1 = -angle1;*/

			v1 = v2;
			v2 = p2 - pointX;

			double angle2 = AngleBetweenVectors(v1, v2);
			/*if (v1.x * v2.y - v1.y * v2.x > 0)
				angle2 = -angle2;*/

			angle1 = abs(angle1);
			angle2 = abs(angle2);
			
			weightSeq[i] = (tan(angle1 / 2.) + tan(angle2 / 2.)) / (Distance(p1, pointX) + 0.001);
			if (weightSeq[i] < 0.0) {
				cout << angle1 << ' ' << angle2 << endl;
				cout << weightSeq[i] << endl;
				system("pause");
			}
			/*if (_isnan(weightSeq[i])) {
				cout << angle1 << ' ' << angle2 << endl;
				printf("%lf, %lf, %lf\n", v1.x, v1.y, v1.z);
				printf("%lf, %lf, %lf\n", v2.x, v2.y, v2.z);
				system("pause");
			}*/
			sum += weightSeq[i];

		}

// 		for (int i = 0; i < size; i++)    
// 		{
// 			Point pt = boundary->at(i);
// 
// 			if (!mask.contains(i)) // if the boundary point is not marked
// 			{
// 				weightSeq[i] = 0;
// 				continue;
// 			}			
// 
// 			p0.x = boundary->at((i - 1 + size) % size).x;
// 			p0.y = boundary->at((i - 1 + size) % size).y;
// 			p0.z = .0;
// 
// 			p1.x = boundary->at(i).x;
// 			p1.y = boundary->at(i).y;
// 			p1.z = 0.00001;
// 
// 			p2.x = boundary->at((i + 1 + size) % size).x;
// 			p2.y = boundary->at((i + 1 + size) % size).y;
// 			p2.z = .0;
// 
// 			v1 = p0 - pointX;
// 			v2 = p1 - pointX;
// 
// 			double angle1 = AngleBetweenVectors(v1, v2);
// 			if (v1.x * v2.y - v1.y * v2.x > 0)
// 				angle1 = -angle1;
// 
// 			v1 = v2;
// 			v2 = p2 - pointX;
// 
// 			double angle2 = AngleBetweenVectors(v1, v2);
// 			if (v1.x * v2.y - v1.y * v2.x > 0)
// 				angle2 = -angle2;
// 
// 			weightSeq[i] = (tan(angle1 / 2.) + tan(angle2 / 2.)) / (Distance(p1, pointX) + 0.001);
// 			sum += weightSeq[i];
// 		}

		return sum;
	}

	void CalculateMVCoord(vector<Point> *boundary, MvcPerporties *mvc, vector<Point> *Coordinates, Mat& mask)
	{
		int BoundarySize = mvc->BoundarySize;
		double *mvcArray = mvc->mvcArray;

		double *weightArray = new double[BoundarySize];

		int vertexSize = Coordinates->size();

		for (int i = 0; i < vertexSize; ++i)
		{
			double sum = CalculateWeightsMask(Point2f(Coordinates->at(i)), boundary, weightArray, mask);
			for (int j = 0; j < BoundarySize; ++j)
				*(mvcArray + i * BoundarySize + j) = (sum > 0.0) ? weightArray[j] / sum : 0.0;			
		}

		delete[] weightArray;
	}

	void CalculateMVCoord(vector<Point> *boundary, MvcPerporties *mvc, vector<Point> *Coordinates, vector<int>& mask)
	{
		int BoundarySize = mvc->BoundarySize;
		double *mvcArray = mvc->mvcArray;

		double *weightArray = new double[BoundarySize];

		int vertexSize = Coordinates->size();

		for (int i = 0; i < vertexSize; ++i)
		{
			double sum = CalculateWeightsVec(Point2f(Coordinates->at(i)), boundary, weightArray, mask);
			for (int j = 0; j < BoundarySize; ++j) {
				*(mvcArray + i * BoundarySize + j) = (abs(sum) >= 1e-12) ? weightArray[j] / sum : 0.0;
				if (*(mvcArray + i * BoundarySize + j) > 1) {
					cout << "> 1" << endl;
					cout << i << ' ' << j << ' ' << weightArray[j] << ' ' << sum << endl;
					cout << *(mvcArray + i * BoundarySize + j) << endl;
					system("pause");
				}
			}
		}
		delete[] weightArray;
	}
}