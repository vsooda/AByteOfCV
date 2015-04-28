#pragma once 
#include "common.h"
#include "windows.h"
const int landmark_num = 74;

int readDir(std::string path, std::vector<std::string> &names, std::string& dir) {
	//vector<string> names;
	dir = path.substr(0, path.find_last_of("\\/") + 1);
	names.clear();
	names.reserve(10000);
	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA(path.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return 0;
	}

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

cv::Mat readPlyData(const char* filename) {
	cv::Mat cloud(1, 1952, CV_32FC3);
	std::ifstream ifs(filename);
	std::string str;
	for (size_t i = 0; i < 14; ++i)
		getline(ifs, str);

	cv::Point3f* data = cloud.ptr<cv::Point3f>();
	int temp1, temp2, temp3, temp4;
	for (size_t i = 0; i < 1952; ++i) {
		ifs >> data[i].x >> data[i].y >> data[i].z >> temp1 >> temp2 >> temp3 >> temp4;
		//std::cout << i + 14 << " " << data[i].x << " " << data[i].y << " " << data[i].z << std::endl;
		//data[i].y = data[i].y - 160.0;
	}
	ifs.close();
	//cloud /= 10.0f;
	return cloud;
}

cv::Mat selectPlyData(const cv::Mat& fullmat, const char* selectIndex) {
	int *index = new int[landmark_num];
	std::ifstream ifs(selectIndex);
	for (int i = 0; i < landmark_num; i++) {
		ifs >> index[i];
	}
	ifs.close();
	cv::Mat selectPt(1, landmark_num, CV_32FC3);
	for (int i = 0; i < landmark_num; i++) {
		selectPt.at<cv::Vec3f>(0, i) = fullmat.at<cv::Vec3f>(0, index[i]);
		//std::cout << index[i] + 14 << " " << selectPt.at<cv::Vec3f>(0, i) << " " << fullmat.at<cv::Vec3f>(0, index[i]) << std::endl;
	}
	delete[] index;
	return selectPt;
}