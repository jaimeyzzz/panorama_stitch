#include "cmfile.h"

#include <shlobj.h>
#include <Commdlg.h>
#include <ShellAPI.h>
#include <omp.h>
#include <fstream>

using namespace std;
using namespace cv;

/************************************************************************/
/* Implementation of inline functions                                   */
/************************************************************************/
string CmFile::GetFolder(const std::string& path)
{
	return path.substr(0, path.find_last_of("\\/") + 1);
}

string CmFile::GetName(const std::string& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_not_of(' ') + 1;
	return path.substr(start, end - start);
}

string CmFile::GetNameNE(const std::string& path)
{
	int start = path.find_last_of("\\/") + 1;
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(start, end - start);
	else
		return path.substr(start, path.find_last_not_of(' ') + 1 - start);
}

string CmFile::GetPathNE(const std::string& path)
{
	int end = path.find_last_of('.');
	if (end >= 0)
		return path.substr(0, end);
	else
		return path.substr(0, path.find_last_not_of(' ') + 1);
}

string CmFile::GetExtention(const std::string name)
{
	return name.substr(name.find_last_of('.'));
}

int CmFile::Copy(const std::string &src, const std::string &dst, int failIfExist)
{
	return ::CopyFileA(src.c_str(), dst.c_str(), failIfExist);
}

int CmFile::Move(const std::string &src, const std::string &dst, DWORD dwFlags)
{
	return MoveFileExA(src.c_str(), dst.c_str(), dwFlags);
}

void CmFile::RmFile(const std::string& fileW)
{
	std::vector<std::string> names;
	string dir;
	int fNum = CmFile::GetNames(fileW, names, dir);
	for (int i = 0; i < fNum; i++)
		::DeleteFileA((dir + names[i]).c_str());
}


// Test whether a file exist
int CmFile::FileExist(const std::string& filePath)
{
	if (filePath.size() == 0)
		return false;

	return  GetFileAttributesA((filePath).c_str()) != INVALID_FILE_ATTRIBUTES; // ||  GetLastError() != ERROR_FILE_NOT_FOUND;
}

int CmFile::FilesExist(const std::string& fileW)
{
	std::vector<std::string> names;
	int fNum = GetNames(fileW, names);
	return fNum > 0;
}

string CmFile::GetWkDir()
{
	string wd;
	wd.resize(1024);
	DWORD len = GetCurrentDirectoryA(1024, &wd[0]);
	wd.resize(len);
	return wd;
}

int CmFile::FolderExist(const std::string& strPath)
{
	int i = (int)strPath.size() - 1;
	for (; i >= 0 && (strPath[i] == '\\' || strPath[i] == '/'); i--)
		;
	std::string str = strPath.substr(0, i + 1);

	WIN32_FIND_DATAA  wfd;
	HANDLE hFind = FindFirstFileA((str).c_str(), &wfd);
	int rValue = (hFind != INVALID_HANDLE_VALUE) && (wfd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
	FindClose(hFind);
	return rValue;
}


int CmFile::MkDir(const string&  _path)
{
	if (_path.size() == 0)
		return FALSE;

	static char buffer[1024];
	strcpy(buffer, (_path).c_str());
	for (int i = 0; buffer[i] != 0; i++) {
		if (buffer[i] == '\\' || buffer[i] == '/') {
			buffer[i] = '\0';
			CreateDirectoryA(buffer, 0);
			buffer[i] = '/';
		}
	}
	return CreateDirectoryA((_path).c_str(), 0);
}


string CmFile::BrowseFolder()
{
	static char Buffer[MAX_PATH];
	BROWSEINFOA bi;//Initial bi 	
	bi.hwndOwner = NULL;
	bi.pidlRoot = NULL;
	bi.pszDisplayName = Buffer; // Dialog can't be shown if it's NULL
	bi.lpszTitle = "BrowseFolder";
	bi.ulFlags = 0;
	bi.lpfn = NULL;
	bi.iImage = NULL;


	LPITEMIDLIST pIDList = SHBrowseForFolderA(&bi); // Show dialog
	if (pIDList)	{
		SHGetPathFromIDListA(pIDList, Buffer);
		if (Buffer[strlen(Buffer) - 1] == '\\')
			Buffer[strlen(Buffer) - 1] = 0;

		return string(Buffer);
	}
	return string();
}

string CmFile::BrowseFile(const char* strFilter, int isOpen)
{
	static char Buffer[MAX_PATH];
	OPENFILENAMEA   ofn;
	memset(&ofn, 0, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.lpstrFile = Buffer;
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = MAX_PATH;
	ofn.lpstrFilter = strFilter;
	ofn.nFilterIndex = 1;
	ofn.Flags = OFN_PATHMUSTEXIST;

	if (isOpen) {
		ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
		GetOpenFileNameA(&ofn);
		return Buffer;
	}

	GetSaveFileNameA(&ofn);
	return string(Buffer);

}

int CmFile::Rename(const string& _srcNames, const string& _dstDir, const char *nameCommon, const char *nameExt)
{
	vector<string> names;
	string inDir;
	int fNum = GetNames(_srcNames, names, inDir);
	for (int i = 0; i < fNum; i++) {
		string dstName = format("%s\\%.4d%s.%s", (_dstDir).c_str(), i, nameCommon, nameExt);
		string srcName = inDir + names[i];
		::CopyFileA(srcName.c_str(), dstName.c_str(), FALSE);
	}
	return fNum;
}

void CmFile::RmFolder(const string& dir)
{
	CleanFolder(dir);
	if (FolderExist(dir))
		RunProgram("Cmd.exe", format("/c rmdir /s /q \"%s\"", (dir).c_str()), true, false);
}

void CmFile::CleanFolder(const string& dir, int subFolder)
{
	vector<string> names;
	int fNum = CmFile::GetNames(dir + "/*.*", names);
	for (int i = 0; i < fNum; i++)
		RmFile(dir + "/" + names[i]);

	vector<string> subFolders;
	int subNum = GetSubFolders(dir, subFolders);
	if (subFolder)
	for (int i = 0; i < subNum; i++)
		CleanFolder(dir + "/" + subFolders[i], true);
}

int CmFile::GetSubFolders(const string& folder, vector<string>& subFolders)
{
	subFolders.clear();
	WIN32_FIND_DATAA fileFindData;
	string nameWC = folder + "\\*";
	HANDLE hFind = ::FindFirstFileA(nameWC.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do {
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			subFolders.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)subFolders.size();
}

// Get image names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
int CmFile::GetNames(const string &nameW, vector<string> &names, string &dir)
{
	dir = GetFolder(nameW);
	names.clear();
	names.reserve(6000);
	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA((nameW).c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE)
		return 0;

	do{
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders
		names.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)names.size();
}

int CmFile::GetNames(const string& rootFolder, const string &fileW, vector<string> &names)
{
	GetNames(rootFolder + fileW, names);
	vector<string> subFolders, tmpNames;
	int subNum = CmFile::GetSubFolders(rootFolder, subFolders);
	for (int i = 0; i < subNum; i++){
		subFolders[i] += "/";
		int subNum = GetNames(rootFolder + subFolders[i], fileW, tmpNames);
		for (int j = 0; j < subNum; j++)
			names.push_back(subFolders[i] + tmpNames[j]);
	}
	return (int)names.size();
}

int CmFile::GetNamesNE(const string& nameWC, vector<string> &names, string &dir, string &ext)
{
	int fNum = GetNames(nameWC, names, dir);
	ext = GetExtention(nameWC);
	for (int i = 0; i < fNum; i++)
		names[i] = GetNameNE(names[i]);
	return fNum;
}

int CmFile::GetNamesNE(const string& rootFolder, const string &fileW, vector<string> &names)
{
	int fNum = GetNames(rootFolder, fileW, names);
	int extS = GetExtention(fileW).size();
	for (int i = 0; i < fNum; i++)
		names[i].resize(names[i].size() - extS);
	return fNum;
}

// Load mask image and threshold thus noisy by compression can be removed
Mat CmFile::LoadMask(const string& fileName)
{
	Mat mask = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	CV_Assert(mask.data != NULL, ("Can't find mask image: %s", (fileName).c_str()));
	compare(mask, 128, mask, CV_CMP_GT);
	return mask;
}

int CmFile::Move2Dir(const string &srcW, const string dstDir)
{
	vector<string> names;
	string inDir;
	int fNum = CmFile::GetNames(srcW, names, inDir);
	int r = TRUE;
	for (int i = 0; i < fNum; i++)
	if (Move(inDir + names[i], dstDir + names[i]) == FALSE)
		r = FALSE;
	return r;
}

int CmFile::Copy2Dir(const string &srcW, const string dstDir)
{
	vector<string> names;
	string inDir;
	int fNum = CmFile::GetNames(srcW, names, inDir);
	int r = TRUE;
	for (int i = 0; i < fNum; i++)
	if (Copy(inDir + names[i], dstDir + names[i]) == FALSE)
		r = FALSE;
	return r;
}

void CmFile::ChkImgs(const string &imgW)
{
	vector<string> names;
	string inDir;
	int imgNum = GetNames(imgW, names, inDir);
	printf("Checking %d images: %s\n", imgNum, (imgW).c_str());
	for (int i = 0; i < imgNum; i++){
		Mat img = imread(inDir + names[i]);
		if (img.data == NULL)
			printf("Loading file %s failed\t\t\n", (names[i]).c_str());
		if (i % 200 == 0)
			printf("Processing %2.1f%%\r", (i*100.0) / imgNum);
	}
	printf("\t\t\t\t\r");
}


void CmFile::RunProgram(const string &fileName, const string &parameters, int waiteF, int showW)
{
	string runExeFile = fileName;
#ifdef _DEBUG
	runExeFile.insert(0, "..\\Debug\\");
#else
	runExeFile.insert(0, "..\\Release\\");
#endif // _DEBUG
	if (!CmFile::FileExist((runExeFile).c_str()))
		runExeFile = fileName;

	SHELLEXECUTEINFOA  ShExecInfo = { 0 };
	ShExecInfo.cbSize = sizeof(SHELLEXECUTEINFO);
	ShExecInfo.fMask = SEE_MASK_NOCLOSEPROCESS;
	ShExecInfo.hwnd = NULL;
	ShExecInfo.lpVerb = NULL;
	ShExecInfo.lpFile = (runExeFile).c_str();
	ShExecInfo.lpParameters = (parameters).c_str();
	ShExecInfo.lpDirectory = NULL;
	ShExecInfo.nShow = showW ? SW_SHOW : SW_HIDE;
	ShExecInfo.hInstApp = NULL;
	ShellExecuteExA(&ShExecInfo);

	//CmLog::LogLine("Run: %s %s\n", ShExecInfo.lpFile, ShExecInfo.lpParameters);

	if (waiteF)
		WaitForSingleObject(ShExecInfo.hProcess, INFINITE);
}

void CmFile::SegOmpThrdNum(double ratio /* = 0.8 */)
{
	int thrNum = omp_get_max_threads();
	int usedNum = cvRound(thrNum * ratio);
	usedNum = max(usedNum, 1);
	//CmLog::LogLine("Number of CPU cores used is %d/%d\n", usedNum, thrNum);
	omp_set_num_threads(usedNum);
}


// Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
void CmFile::copyAddSuffix(const string &srcW, const string &dstDir, const string &dstSuffix)
{
	vector<string> namesNE;
	string srcDir, srcExt;
	int imgN = CmFile::GetNamesNE(srcW, namesNE, srcDir, srcExt);
	CmFile::MkDir(dstDir);
	for (int i = 0; i < imgN; i++)
		CmFile::Copy(srcDir + namesNE[i] + srcExt, dstDir + namesNE[i] + dstSuffix);
}

vector<string> CmFile::loadStrList(const string &fName)
{
	ifstream fIn(fName);
	string line;
	vector<string> strs;
	while (getline(fIn, line) && line.size())
		strs.push_back(line);
	return strs;
}

int CmFile::writeStrList(const string &fName, const vector<string> &strs)
{
	FILE *f = fopen((fName).c_str(), "w");
	if (f == NULL)
		return false;
	for (size_t i = 0; i < strs.size(); i++)
		fprintf(f, "%s\n", (strs[i]).c_str());
	fclose(f);
	return true;
}