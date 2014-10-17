#ifndef OPENCV_INPAINT_H_H
#define OPENCV_INPAINT_H_H


#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <iostream>

namespace customCV {


	void inpaint(cv::Mat src, cv::Mat inpaint_mask, cv::Mat& dst, int range);

	//template<typename T>
	//T getElement(T i,const cv::Mat& mat, int row, int col, int color);
}

#endif // !OPENCV_INPAINT_H_H