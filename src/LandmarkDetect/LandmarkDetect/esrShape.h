#include "dlib/image_processing.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "dlib/opencv.h"
#include "common.h"

typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;

class EsrShape {
public:
	EsrShape(const char* faceName, const char* shapeName, int landmarkNum = 74);
	~EsrShape();
	void  detect(const cv::Mat& src);
	void  draw(cv::Mat& src);
	std::vector<cv::Point2f> getFilterPts();
private:
	frontal_face_detector _detector;
	customCV::shape_predictor _sp;
	cv::Rect _rect;
	std::vector<cv::Point2f> _pts;
	int _landnum;
};

EsrShape::EsrShape(const char* faceName, const char* shapeName, int landmarkNum)
{
	_landnum = landmarkNum;
	dlib::deserialize(faceName) >> _detector;
	//dlib::deserialize(shapeName) >> _sp;
	_sp = load_ft<customCV::shape_predictor>(shapeName);
}

EsrShape::~EsrShape() {
}

void  EsrShape::detect(const cv::Mat& src)
{
	_pts.clear();
	dlib::array2d<dlib::rgb_pixel> img;
	std::vector<cv::Rect> faces;
	dlib::cv_image<dlib::rgb_pixel> *pimg = new dlib::cv_image<dlib::rgb_pixel>(src);
	assign_image(img, *pimg);

	std::vector<dlib::rectangle> dets;
	dets = _detector(img);
	if (dets.size() <= 0) {
		return;
	}
	_rect = cv::Rect(cv::Point(dets[0].left(), dets[0].top()), cv::Point(dets[0].right(), dets[0].bottom()));
	dlib::rectangle det(_rect.x, _rect.y, _rect.x + _rect.width, _rect.y + _rect.height);
	cv::Mat avg(src.size(), CV_32FC1, cv::Scalar(0));
	for (int x = 0; x < avg.cols; x++) {
		for (int y = 0; y < avg.rows; y++) {
			cv::Vec3b temp = src.at<cv::Vec3b>(y, x);
			avg.at<float>(y, x) = temp[0] / 3 + temp[1] / 3 + temp[2] /3 ;
		}
	}
	std::cout << "detect ok" << std::endl;
	cv::Mat shape = _sp(avg, _rect);
	cv::Mat currentMat(2, _landnum, CV_32F);
	for (int k = 0; k < _landnum; k++) {
		currentMat.at<float>(0, k) = shape.at<float>(2 * k, 0);
		currentMat.at<float>(1, k) = shape.at<float>(2 * k + 1, 0);
	}
	AffineTransform tform_to_img = customCV::impl::unnormalizing_tform(_rect);
	currentMat = tform_to_img(currentMat);
	for (int i = 0; i < currentMat.cols; i++) {
		_pts.push_back(cv::Point2f(currentMat.at<float>(0, i), currentMat.at<float>(1, i)));
	}
	delete pimg;
}


void  EsrShape::draw(cv::Mat& src) {
	for (int i = 0; i < _pts.size(); i++) {
		cv::circle(src, _pts[i], 3, cv::Scalar(255, 0, 255), -1);
	}
	cv::rectangle(src, _rect, cv::Scalar(0, 255, 0), 2);
}

std::vector<cv::Point2f> EsrShape::getFilterPts() {
	if (_landnum == 74) {
		std::vector<cv::Point2f> pts;
		for (int i = 0; i < 68; i++) {
			if (i <= 30) {
				pts.push_back(_pts[i]);
			}
			else if (i == 31) {
				float x = (_pts[27].x + _pts[29].x) / 2;
				float y = (_pts[28].y + _pts[30].y) / 2;
				pts.push_back(cv::Point2f(x, y));
			}
			else if (i < 35) {
				pts.push_back(_pts[i - 1]);
			}
			else if (i == 36) {
				float x = (_pts[32].x + _pts[34].x) / 2;
				float y = (_pts[33].y + _pts[35].y) / 2;
				pts.push_back(cv::Point2f(x, y));
			}
			else {
				pts.push_back(_pts[i - 2]);
			}
		}
		return pts;
	}
	else {
		return _pts;
	}
}