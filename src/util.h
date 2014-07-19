#ifndef __OPENCV_UUTIL_H__ 
#define __OPENCV_UUTIL_H__

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgproc/imgproc_c.h"
#include <vector>
#include <iostream>
#include <stdarg.h>

namespace customCV {
	//IplImage* Transform(IplImage* A,CvScalar avg_src, CvScalar avg_dst,CvScalar std_src, CvScalar std_dst);
	IplImage* Transform(IplImage* A,CvScalar avg_src, CvScalar avg_dst,CvScalar std_src, CvScalar std_dst, IplImage* mask)  ;	
	void cvShowManyImages(char* title, int nArgs, ...);

	cv::Mat  Dilation(cv::Mat src, int size = 1, int type = 0);
	cv::Mat Erosion(cv::Mat src, int size = 1 , int type = 0);
	void LBP (IplImage *src,IplImage *dst) ;
	//void  transformMat(cv::Mat& src, cv::Scalar avg_src, cv::Scalar avg_dst, cv::Scalar std_src, cv::Scalar std_dst, cv::Mat mask);
	void  transformMat(cv::Mat& src, cv::Scalar avg_src, cv::Scalar avg_dst, cv::Scalar std_src, cv::Scalar std_dst, cv::InputArray mask = cv::noArray());

	struct funcParam {
		float val[10];
	};


	//template <typename T>
	//void matFilter(cv::Mat& mat, T minV, T maxV);
	template <typename T>
	void matFilter(cv::Mat& mat, T minV, T maxV) {
		int rows = mat.rows;
		int cols = mat.cols;
		int channels = mat.channels();
		int count = 0;
		if(channels == 1) {
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					T temp = mat.at<T>(i, j);
					temp = temp < minV? minV : temp;
					temp = temp > maxV? maxV : temp;
					mat.at<T>(i, j) = temp;
					count ++;
				}
			}	
		}
		else if(channels == 3) {
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					for(int c = 0; c < 3; c++) {
						T temp = mat.at<cv::Vec<int, 3> >(i, j)[c];
						//temp = temp < minV? minV : temp;
						//temp = temp > maxV? maxV : temp;
						if(temp < minV) {
							minV = 0.0;
							count++;
						}
						else if(temp > maxV) {
							maxV = 255.0;
							count++;
						}
						mat.at<cv::Vec<int, 3> >(i, j)[c] = temp;
						count ++;
					}
				}
			}
		}
		std::cout << count << std::endl;
		
	}

}


#endif