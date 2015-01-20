#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)


#include <assert.h>
#include <string>
#include <xstring>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <exception>
#include <cmath>
#include <time.h>
#include <set>
#include <queue>
#include <list>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
#include <atlstr.h>
#include <atltypes.h>
#include <omp.h>
#include <strstream>
#include <stdarg.h>
#include <string>
#include <windows.h>


#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif


#include <opencv2/opencv.hpp> 
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)

#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
#pragma comment( lib, cvLIB("photo"))
#pragma comment( lib, cvLIB("viz"))
#pragma comment( lib, cvLIB("calib3d"))

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


// ----------------------------------------------------------------------------------------

template <class T>
T load_ft(const char* fname){
	T x; cv::FileStorage f(fname, cv::FileStorage::READ);
	f["ft object"] >> x; f.release(); return x;
}
//==============================================================================
template<class T>
void save_ft(const char* fname, const T& x){
	cv::FileStorage f(fname, cv::FileStorage::WRITE);
	f << "ft object" << x; f.release();
}
