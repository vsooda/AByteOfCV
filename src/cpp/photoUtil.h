#pragma once
#include <iostream>
#include <string>
#include <ctime>
#include <algorithm>
#include <functional>
#include <opencv2/opencv.hpp> 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)

#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
#pragma comment( lib, cvLIB("photo"))


namespace customCV {
	//膨胀
	cv::Mat  Dilation(cv::Mat src, int size = 1, int type = 0);

	//腐蚀
	cv::Mat Erosion(cv::Mat src, int size = 1, int type = 0);

	//相似性采样
	cv::Point2i samplePos(cv::Mat tmplate, cv::Mat DICT_IM_SQUARED, cv::Mat DICT_IM);

	//美白
	void  whiteSkin(const cv::Mat &src, cv::Mat& dst, int beta = 5);
	void whiteSkinC1(const cv::Mat& src, cv::Mat& dst, int beta = 5);

	//颜色传输
	cv::Mat colorTransform(cv::Mat src, cv::Mat dst, cv::InputArray maskMat = cv::noArray());

}
