#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <windows.h>

struct CmFile
{
	static std::string BrowseFile(const char* strFilter = "Images (*.jpg;*.png)\0*.jpg;*.png\0All (*.*)\0*.*\0\0", int isOpen = true);
	static std::string BrowseFolder();

	static inline std::string GetFolder(const std::string& path);
	static inline std::string GetName(const std::string& path);
	static inline std::string GetNameNE(const std::string& path);
	static inline std::string GetPathNE(const std::string& path);

	// Get file names from a wildcard. Eg: GetNames("D:\\*.jpg", imgNames);
	static int GetNames(const std::string &nameW, std::vector<std::string> &names, std::string &dir = std::string());
	static int GetNames(const std::string& rootFolder, const std::string &fileW, std::vector<std::string> &names);
	static int GetNamesNE(const std::string& nameWC, std::vector<std::string> &names, std::string &dir = std::string(), std::string &ext = std::string());
	static int GetNamesNE(const std::string& rootFolder, const std::string &fileW, std::vector<std::string> &names);
	static inline std::string GetExtention(const std::string name);

	static inline int FileExist(const std::string& filePath);
	static inline int FilesExist(const std::string& fileW);
	static inline int FolderExist(const std::string& strPath);

	static inline std::string GetWkDir();

	static int MkDir(const std::string&  path);

	// Eg: RenameImages("D:/DogImages/*.jpg", "F:/Images", "dog", ".jpg");
	static int Rename(const std::string& srcNames, const std::string& dstDir, const char* nameCommon, const char* nameExt);

	static inline void RmFile(const std::string& fileW);
	static void RmFolder(const std::string& dir);
	static void CleanFolder(const std::string& dir, int subFolder = false);

	static int GetSubFolders(const std::string& folder, std::vector<std::string>& subFolders);

	inline static int Copy(const std::string &src, const std::string &dst, int failIfExist = FALSE);
	inline static int Move(const std::string &src, const std::string &dst, DWORD dwFlags = MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED | MOVEFILE_WRITE_THROUGH);
	static int Move2Dir(const std::string &srcW, const std::string dstDir);
	static int Copy2Dir(const std::string &srcW, const std::string dstDir);

	//Load mask image and threshold thus noisy by compression can be removed
	static cv::Mat LoadMask(const std::string& fileName);

	static void WriteNullFile(const std::string& fileName) { FILE *f = fopen((fileName).c_str(), "w"); fclose(f); }

	static void ChkImgs(const std::string &imgW);

	static void RunProgram(const std::string &fileName, const std::string &parameters = "", int waiteF = false, int showW = true);

	static void SegOmpThrdNum(double ratio = 0.8);

	// Copy files and add suffix. e.g. copyAddSuffix("./*.jpg", "./Imgs/", "_Img.jpg")
	static void copyAddSuffix(const std::string &srcW, const std::string &dstDir, const std::string &dstSuffix);

	static std::vector<std::string> loadStrList(const std::string &fName);
	static int writeStrList(const std::string &fName, const std::vector<std::string> &strs);
};