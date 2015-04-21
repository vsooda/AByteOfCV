#ifndef ESR_SHAPE_H
#define ESR_SHAPE_H
#include "dlib/image_processing.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "dlib/opencv.h"
#include "common.h"
#include "AffineTransform.h"

typedef dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > > frontal_face_detector;

class EsrShape {
public:
	EsrShape(const char* faceName, const char* shapeName, int landmarkNum = 74);
	~EsrShape();
	bool  detect(const cv::Mat& src);
	void  draw(cv::Mat& src);
	cv::Mat getInitShape();  //1 * landmark * 2 
	std::vector<cv::Point2f> getFilterPts();
	std::vector<cv::Point2f> getPts();
	cv::Rect getRect();
private:
	frontal_face_detector _detector;
	customCV::shape_predictor _sp;
	dlib::shape_predictor _dlib_sp;
	cv::Rect _rect;
	std::vector<cv::Point2f> _pts;
	cv::Mat _initShape;
	int _landnum;
	int _dlibtype;
};

EsrShape::EsrShape(const char* faceName, const char* shapeName, int landmarkNum)
{
	_dlibtype = 0;
	_landnum = landmarkNum;
	dlib::deserialize(faceName) >> _detector;
	//dlib::deserialize(shapeName) >> _sp;
	if (_dlibtype == 0) {
		dlib::deserialize(shapeName) >> _dlib_sp;
	}
	else {
		_sp = load_ft<customCV::shape_predictor>(shapeName);
	}
}

EsrShape::~EsrShape() {
}

bool EsrShape::detect(const cv::Mat& src)
{
	_pts.clear();
	dlib::array2d<dlib::rgb_pixel> img;
	std::vector<cv::Rect> faces;
	dlib::cv_image<dlib::rgb_pixel> *pimg = new dlib::cv_image<dlib::rgb_pixel>(src);
	assign_image(img, *pimg);

	std::vector<dlib::rectangle> dets;
	dets = _detector(img);
	if (dets.size() <= 0) {
		return false;
	}
	_rect = cv::Rect(cv::Point(dets[0].left(), dets[0].top()), cv::Point(dets[0].right(), dets[0].bottom()));
	dlib::rectangle det(_rect.x, _rect.y, _rect.x + _rect.width, _rect.y + _rect.height);
	if (_dlibtype == 0) {
		dlib::full_object_detection shape = _dlib_sp(img, det);
		for (int i = 0; i < shape.num_parts(); i++) {
			dlib::point pt = shape.part(i);
			_pts.push_back(cv::Point2f(pt.x(), pt.y()));
		}
	}
	else {
		cv::Mat avg(src.size(), CV_32FC1, cv::Scalar(0));
		for (int x = 0; x < avg.cols; x++) {
			for (int y = 0; y < avg.rows; y++) {
				cv::Vec3b temp = src.at<cv::Vec3b>(y, x);
				avg.at<float>(y, x) = temp[0] / 3 + temp[1] / 3 + temp[2] / 3;
			}
		}
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
	}
	delete pimg;
	return true;
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
			else if (i <= 35) {
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

std::vector<cv::Point2f> EsrShape::getPts() {
	return _pts;
}

cv::Rect EsrShape::getRect() {
	return _rect;
}

cv::Mat EsrShape::getInitShape() {
	CV_Assert(_dlibtype == 1);
	return _sp.getInitShape();
}

#endif