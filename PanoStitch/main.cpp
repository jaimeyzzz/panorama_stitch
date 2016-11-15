#include <vector>
#include <sstream>
#include <ctime>
#include <opencv2\opencv.hpp>

#include "FeatherStitch.h"
#include "MVCStitch.h"
#include <windows.h>
#include <psapi.h>

using namespace cv;
using namespace std;

string sample = "Z:\\ljm\\PanoStitch_sample\\scenetest\\";

string scriptFile = sample + "script.txt";
string inputPath = sample;
string videoFormat = "mp4";
string outputPath = "res\\";
vector<int> delay = { 0,0, 0, 0, 0, 0 };
vector<int> order = { 0, 4, 5,3, 1, 2 };
//vector<int> order = { 3, 4, 5, 2, 0, 1 };
int startFrameNumber =140;
int endFrameNumber = 140;

void genTemplate(string videodir, string format, int imNum, Size sz);

void featherStitch() {
	FeatherStitch stitcher(scriptFile);
	stitcher.init(inputPath, true, videoFormat);
	stitcher.setDelay(delay);
	stitcher.setOrder(order);
	stitcher.setStartFrameNumber(startFrameNumber);
	stitcher.setEndFrameNumber(endFrameNumber);
	stitcher.stitch(outputPath);
}

void mvcStitch() {
	MVCStitch stitcher(scriptFile);
	stitcher.setDelay(delay);
	stitcher.setOrder(order);
	stitcher.setStartFrameNumber(startFrameNumber);
	stitcher.setEndFrameNumber(endFrameNumber);

	stitcher.init(inputPath, true, videoFormat);
	stitcher.stitch(outputPath);
}
void showMemoryInfo(void)
{
	HANDLE handle = GetCurrentProcess();
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(handle, &pmc, sizeof(pmc));
	cout << "ÄÚ´æÊ¹ÓÃ:" << pmc.WorkingSetSize / 1000 << "K/" << pmc.PeakWorkingSetSize / 1000 << "K + " << pmc.PagefileUsage / 1000 << "K/" << pmc.PeakPagefileUsage / 1000 << "K" << endl;
	//cout << PSAPI_VERSION;
}
int main() {
	//genTemplate(sample + "\\", "avi", 6, Size(1440, 1080));
	featherStitch();
	//mvcStitch();

	/*MultiBandStitch stitcher(sample + "\\script.txt");
	stitcher.init(sample + "\\", true, "mp4");
	stitcher.stitch(sample + "\\multiband.avi");*/
	showMemoryInfo();
	system("pause");
	return 0;
}

void genTemplate(string videodir, string format, int imNum, Size sz) {
	int turns = 800;
	vector<VideoCapture> videos;

	videos.resize(imNum);
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << videodir << i << "." << format;
		if (!videos[i].open(ostr.str())) {
			cout << "Load video error" << endl;
			system("pause");
			exit(0);
		}
	}
	for (int i = 0; i < imNum; i++) {
		ostringstream ostr;
		ostr << videodir << i << ".png";
		Mat frame;
		for (int j = 0; j < turns; j++) {
			if (!videos[i].read(frame)) {
				system("pause");
			}
		}
		resize(frame, frame, sz);
		imwrite(ostr.str(), frame);
	}
	exit(0);
}

