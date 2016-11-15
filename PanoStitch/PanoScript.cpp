/*************************************************
* PanoScript类
* 读取PanoStitcher脚本，得到remapping表对N路视频流进行变换。
*************************************************/

#include "PanoScript.h"
#include "PanoScriptTransform.h"
#include "CmFile.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cmath>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


/** 构造函数输入脚本文件路径，读入脚本文件 */
PanoScript::PanoScript(string scriptName) {
	if (CmFile::FileExist(scriptName))
		loadScript(scriptName);
	else {
		printf("File %s not exist\n", scriptName.c_str());
		system("pause");
		exit(0);
	}
}

PanoScript::~PanoScript() {
}

/** 读入PanoScript脚本文件的p行 */
void PanoScript::readPanoLine(string line) {
	istringstream istr;
	istr.str(line);
	string str;
	istr >> str;
	while (!istr.eof()) {
		istr >> str;
		switch (str[0]) {
		case 'w':
			sscanf(str.c_str(), "w%d", &pano.w);
			break;
		case 'h':
			sscanf(str.c_str(), "h%d", &pano.h);
			break;
		case 'v':
			sscanf(str.c_str(), "v%lf", &pano.hfov);
			break;
		default:
			break;
		}
	}
}

/** 读入PanoScript脚本文件的i或o行 */
void PanoScript::readImageLine(string line, int idx) {
	// read 'o' lines;
	// ignore '+buff', '-buff', 'u'.etc(obsolete) and 'o'
	istringstream istr;
	istr.str(line);
	string str;
	istr >> str;
	while (!istr.eof()) {
		istr >> str;
		switch (str[0]) {
		case 'w':
			sscanf(str.c_str(), "w%d", &(images[idx].w));
			break;
		case 'h':
			sscanf(str.c_str(), "h%d", &(images[idx].h));
			break;
		case 'f':
			sscanf(str.c_str(), "f%d", &(images[idx].f));
			break;
		case 'v':
			sscanf(str.c_str(), "v%lf", &(images[idx].hfov));
			break;
		case 'r':
			sscanf(str.c_str(), "r%lf", &(images[idx].roll));
			break;
		case 'p':
			sscanf(str.c_str(), "p%lf", &(images[idx].pitch));
			break;
		case 'y':
			sscanf(str.c_str(), "y%lf", &(images[idx].yaw));
			break;
		case 'a':
			sscanf(str.c_str(), "a%lf", &(images[idx].rad[3]));
			images[idx].radial = true;
			break;
		case 'b':
			sscanf(str.c_str(), "b%lf", &(images[idx].rad[2]));
			images[idx].radial = true;
			break;
		case 'c':
			sscanf(str.c_str(), "c%lf", &(images[idx].rad[1]));
			images[idx].radial = true;
			break;
		case 'd':
			sscanf(str.c_str(), "d%lf", &(images[idx].hori));
			images[idx].horizontal = true;
			break;
		case 'e':
			sscanf(str.c_str(), "e%lf", &(images[idx].vert));
			images[idx].vertical = true;
			break;
		case 'S':
			sscanf(str.c_str(), "S%d,%d,%d,%d", &(images[idx].S[0]), &(images[idx].S[1]), &(images[idx].S[2]), &(images[idx].S[3]));
			images[idx].select = true;
			break;
		case 'C':
			sscanf(str.c_str(), "C%d,%d,%d,%d", &(images[idx].C[0]), &(images[idx].C[1]), &(images[idx].C[2]), &(images[idx].C[3]));
			images[idx].crop = true;
			break;
		default:
			break;
		}
	}
	images[idx].rad[0] = 1.0 - (images[idx].rad[1] + images[idx].rad[2] + images[idx].rad[3]);
	SetCorrectionRadius(images[idx].rad);
}


/** 读入PanoScript脚本 */
void PanoScript::loadScript(string scriptName) {
	ifstream fin;
	fin.open(scriptName);
	int image_idx = 0;
	while (!fin.eof()) {
		string line;
		getline(fin, line);
		switch (line[0]) {
		case 'p' :
			readPanoLine(line);
			break;
		case 'i':
			images.push_back(ImageParam());
			//ignore n parameter and i lines;
			//readImageLine(line, image_idx);
			break;
		case 'o':
			readImageLine(line, image_idx ++);
			break;
		default:
			break;
		}
	}
	fin.close();
#if 0 // print out pano info for debugging
	cout << "****** pano info ******" << endl;
	cout << "w : " << pano.w << " h : " << pano.h << endl;
	cout << "v : " << pano.hfov << endl;
#endif
}

// TODO:
void PanoScript::getROI() {}

/** 设置某一帧对应的变换参数 */
void PanoScript::setTransPram(int idx) {
	curImagePtr = &images[idx];
	ImageParam& image = images[idx]; // 设置当前图像
	
	if (image.crop) { // 如果需要裁剪，根据裁剪区域更新图像的宽度和高度
		image.w = image.C[1] - image.C[0];
		image.h = image.C[3] - image.C[2];
	}

	int image_selection_width = image.w;
	int image_selection_height = image.h;

	/*if (image.crop) {
		
		image_selection_width = image.C[1] - image.C[0];
		image_selection_height = image.C[3] - image.C[2];
		trans.horizontal += (image.C[1] + image.C[0] - image.w) / 2.0;
		trans.vertical += (image.S[3] + image.S[2] - image.h) / 2.0;
	}*/
	//cout << image.w << ' ' << image.h << ' ' << image.f << endl;

	// 全景图和某路帧对应的视角
	double a = DEG_TO_RAD(image.hfov);
	double b = DEG_TO_RAD(pano.hfov);

	// 设置变换矩阵
	SetMatrix(-DEG_TO_RAD(image.pitch),
		0.0,
		-DEG_TO_RAD(image.roll),
		trans.mt,
		0);

	trans.distance = ((double)pano.w) / b;
	// 根据图像类型，计算缩放参数
	switch (image.f) {
	case _rectilinear:
		// calculate distance for this projection
		trans.scale[0] = (double)image_selection_width / (2.0 * tan(a / 2.0)) / trans.distance;
		break;
	case _equirectangular:
	case _panorama:
	case _fisheye_ff:
	case _fisheye_circ:
		trans.scale[0] = ((double)image_selection_width) / a / trans.distance;
		break;
	default:
		cout << "SetMakeParams: Unsupported input image projection" << endl;
		system("pause");
		exit(0);
	}
	trans.scale[1] = trans.scale[0];

	trans.rot[0] = trans.distance * PI;                                // 180 in screenpoints
	trans.rot[1] = - image.yaw * trans.distance * PI / 180.0;            // rotation angle in screenpoints

	// 透视变换参数
	trans.perspect[0] = (void*)(trans.mt);
	trans.perspect[1] = (void*)&(trans.distance);

	// 镜像畸变校正参数
	for (int i = 0; i < 4; i ++)
		trans.rad[i] = image.rad[i];
	trans.rad[5] = image.rad[4];
	trans.rad[4] = ((double)image.h) / 2.0;
	// 调节水平竖直偏移
	if (image.horizontal) trans.horizontal = image.hori;
	else trans.horizontal = 0.0;
	if (image.vertical) trans.vertical = image.vert;
	else trans.vertical = 0.0;
}

/** 通过变换参数，对每一个像素点进行变换，由全景图像位置对应到每路帧的位置 */
void PanoScript::transform(double x, double y, double* dx, double* dy) {
	// use params in trans to transform x, y
	// TODO : plane_transfer_to_camera?

	rotate_erect(x, y, dx, dy, trans.rot); // Rotate equirect. image horizontally
	sphere_tp_erect(*dx, *dy, dx, dy, &(trans.distance)); // Convert spherical image to equirect.
	persp_sphere(*dx, *dy, dx, dy, trans.perspect); // Pers0pective Control spherical Image

	switch (curImagePtr->f) {
	case _rectilinear:
		rect_sphere_tp(*dx, *dy, dx, dy, &(trans.distance));
		break;
	case _panorama:                                   //  pamoramic image
		pano_sphere_tp(*dx, *dy, dx, dy, &(trans.distance)); // Convert spherical to pano
		break;
	case _equirectangular:                       //  equirectangular image
		erect_sphere_tp(*dx, *dy, dx, dy, &(trans.distance)); // Convert spherical to equirect
		break;
	case _fisheye_circ:
	case _fisheye_ff:
		; // no conversion needed. It is already in spherical coordinates
		break;
	default:
		cout << "Invalid input projection " << curImagePtr->f << ". Assumed fisheye." << endl;
		system("pause");
		exit(0);
	}
	resize(*dx, *dy, dx, dy, trans.scale);
	radial(*dx, *dy, dx, dy, trans.rad);
	
	if (trans.horizontal != 0.0) {
		horiz(*dx, *dy, dx, dy, &(trans.horizontal));
	}
	if (trans.vertical != 0.0) {
		vert(*dx, *dy, dx, dy, &(trans.vertical));
	}
}

PanoScript::ImageParam::ImageParam() {
	f = w = h = 0;
	hori = vert = roll = pitch = yaw = hfov = 0.0;
	memset(rad, 0, sizeof(double) * 6);
	memset(S, 0, sizeof(int)* 4);
	memset(C, 0, sizeof(int)* 4);
	rad[0] = 1.0;
	rad[4] = 1000.0;
	select = crop = radial = horizontal = vertical = false;
	pano = NULL;
}