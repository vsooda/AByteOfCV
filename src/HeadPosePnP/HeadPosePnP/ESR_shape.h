#ifndef ESR_SHAPE_H_H
#define ESR_SHAPE_H_H 
#include "../../dlib/image_processing.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "../../dlib/opencv.h"
#include "AffineTransform.h"

typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;

class EsrShape {
public:
	EsrShape(const char* faceName, const char* shapeName, int landmarkNum = 74);
	~EsrShape();
	bool  detect(const cv::Mat& src);
	void detectWithRect(const cv::Mat&  src, cv::Rect rect);
	void detectAndRotateWithRect(const cv::Mat&  src, cv::Rect rect, cv::Mat& rotateImage);
	void  draw(cv::Mat& src);
	std::vector<cv::Point2f> getFilterPts();
	std::vector<cv::Point2f> getPts5();
	void setInitPts();
	std::vector<cv::Point2f> getAllPts();
	std::vector<cv::Point2f> getRotatePts(std::vector<cv::Point2f> pts, std::vector<cv::Point2f> initPts, float* pangle = NULL);
	std::vector<cv::Point2f> getRotatePts(std::vector<cv::Point2f> queryPts, std::vector<cv::Point2f> pts, std::vector<cv::Point2f> initPts, float* pangle = NULL);
	cv::Mat getRotateMat(const cv::Mat& src, std::vector<cv::Point2f> pts, std::vector<cv::Point2f> initPts, float* pangle = NULL);
	bool detectAndRotate(const cv::Mat& src, cv::Mat& rotatedImage);
	void similarity_transform_correct(std::vector<cv::Point2f>& query_pts, const std::vector<cv::Point2f>& detect_pts, const std::vector<cv::Point2f>& correct_pts);
	std::vector<cv::Point2f> filter74to68(const std::vector<cv::Point2f>& pts);
	void setRotateEstimateIndexs(std::vector<int> indexs);
	void setRotateEstimateIndexs74();
	void setRotateEstimateIndexs68();
	std::vector<cv::Point2f> getIndexPts(const std::vector<cv::Point2f> &pts, std::vector<int> indexs);
	cv::Rect getFaceRect();
	void rotateMatAndPts(cv::Mat& src, std::vector<cv::Point2f> &pts, float* pangle = NULL);
	void indexRotateMatAndPts(cv::Mat& src, std::vector<cv::Point2f> &pts, float* pangle);
private:
	frontal_face_detector _detector;
	dlib::shape_predictor _sp;
	cv::Rect _rect;
	std::vector<cv::Point2f> _pts;
	int _landnum;
	std::vector<cv::Point2f> _initPts;
	std::vector<int> _rotateEstimateIndexs;
};




#endif

/*
int main() {
    EsrShape esp("front_face.dat", "sp_10000.dat");
    cv::Mat img = cv::imread("img_0001.jpg");
    esp.detect(img);
    std::vector<cv::Point2f> pts = esp.getFilterPts();
    for(int i = 0; i < pts.size(); i++) {
        cv::circle(img, pts[i], 3, cv::Scalar(0, 0, 255), -1);
    }
    //esp.draw(img);
    cv::imshow("dst", img);
    cv::waitKey();
}
*/