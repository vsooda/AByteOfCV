#pragma once
#include <vector>
#include <opencv/cv.h> 
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;
 


bool  doCheck_eye(Mat image_in, std::vector<cv::Point2f>& point2f_vector,int upindex,int midindex,int downindex,int conindex1,int conindex2);
bool  doCheck_mouth(Mat image_in, std::vector<cv::Point2f>& point2f_vector,int leftindex,int rightindex,int upindex,int downindex);
std::vector<cv::Point2f> _ReadPTS(const char *filename);
void  _WritePTS(const char *filename,std::vector<cv::Point2f> totalpoint2f);
void  _WriteTXT(const char *filename, std::vector<cv::Point2f> totalpoint2f);


float otsu(Mat A,float fmin,float fmax);
