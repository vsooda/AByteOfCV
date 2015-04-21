#ifndef HEAD_POSE_H_H
#define HEAD_POSE_H_H
//基本是保存3d点，和2d点。以及各种计算方法
#include <iostream>
#include <windows.h>
#include <vector>
#include "common.h"
#include "dlib/opencv.h"
#include <opencv2/viz/vizcore.hpp>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>

class HeadPose {
public:
	HeadPose(const char* plyname, const char* indexName, int landnum = 74) {
		landnum_ = landnum;
	}

	void show2dProject(cv::Mat dstView, cv::Mat srcmat);

	//change vector point  or mat point to the uniform format .
	cv::Mat pts2Mat(const std::vector<cv::Point2f> pts);
	cv::Mat muitiChanelMat2singleChannel(cv::Mat muitiMat);
	cv::Mat singleChanelMat2muitiChannel(cv::Mat singleMat);
	cv::Mat estimate2dRotate(cv::Mat& fromPtmat, cv::Mat& toPtmat, double *pangle = NULL);
	void normalizeRows(cv::Mat& src, int rowIndex = -1);
	cv::Mat tform2image(cv::Mat& ptmat, cv::Rect rect);
	cv::Mat image2tform(cv::Mat& ptmat, cv::Rect rect);
	cv::Mat singleColShape2ptmat(cv::Mat oneColMat);
	cv::Mat ptmat2singleColsShape(cv::Mat& ptmat);
	void resetAngle();
	void showNormalizePtmat(cv::Mat &view, cv::Mat ptmat, cv::Scalar color = cv::Scalar(255, 255, 255));
	void visualize();
	void setInitMat();

	int landnum_;
	cv::Mat detPtMat_;
	cv::Mat tformDetPtmat_;
	cv::Mat rotatePtmat_;
	float angleX_, angleY_, angleZ_;
	cv::Mat image_;
	cv::Mat initShape_;
	cv::Mat initShapePtmat_;
	std::vector<cv::Point2f> pts_;
	cv::Rect rect_;
};


void HeadPose::showNormalizePtmat(cv::Mat &view, cv::Mat ptmat, cv::Scalar color) {
	for (int k = 0; k < ptmat.cols; k++) {
		int x = ptmat.at<float>(0, k) * view.cols;
		int y = ptmat.at<float>(1, k) * view.rows;
		cv::circle(view, cv::Point(x, y), 2, color, -1);
	}
}

//row present the corridate . col present the diffent point
cv::Mat HeadPose::pts2Mat(const std::vector<cv::Point2f> pts) {
	cv::Mat ptmat(2, pts.size(), CV_32FC1);
	for (int i = 0; i < pts.size(); i++) {
		ptmat.at<float>(0, i) = pts[i].x;
		ptmat.at<float>(1, i) = pts[i].y;
	}
	return ptmat;
}

//muitlchannelmat is in the format (1 row, n cols)
//only process CV_32FC3
cv::Mat HeadPose::muitiChanelMat2singleChannel(cv::Mat muitiMat) {
	CV_Assert(muitiMat.type() == CV_32FC3);
	cv::Mat ptmat(3, muitiMat.cols, CV_32FC1);
	for (int i = 0; i < muitiMat.cols; i++) {
		cv::Vec3f temp = muitiMat.at<cv::Vec3f>(0, i);
		ptmat.at<float>(0, i) = temp[0];
		ptmat.at<float>(1, i) = temp[1];
		ptmat.at<float>(2, i) = temp[2];
	}
	return ptmat;
}

cv::Mat HeadPose::singleChanelMat2muitiChannel(cv::Mat singleMat) {
	CV_Assert(singleMat.rows == 3 && singleMat.type() == CV_32FC1);
	cv::Mat muitimat(1, singleMat.cols, CV_32FC3);
	for (int i = 0; i < singleMat.cols; i++) {
		float x = singleMat.at<float>(0, i);
		float y = singleMat.at<float>(1, i);
		float z = singleMat.at<float>(2, i);
		muitimat.at<cv::Vec3f>(0, i) = cv::Vec3f(x, y, z);
	}
	return muitimat;
}

void HeadPose::setInitMat() {
	detPtMat_ = pts2Mat(pts_);
	tformDetPtmat_ = image2tform(detPtMat_, rect_);
}

cv::Mat HeadPose::singleColShape2ptmat(cv::Mat oneColMat) {
	CV_Assert(oneColMat.rows % 2 == 0 && oneColMat.type() == CV_32FC1);
	int ptnums = oneColMat.rows / 2;
	cv::Mat ptmat(2, ptnums, CV_32FC1);
	for (int k = 0; k < landnum_; k++) {
		ptmat.at<float>(0, k) = oneColMat.at<float>(2 * k, 0);
		ptmat.at<float>(1, k) = oneColMat.at<float>(2 * k + 1, 0);
	}
	return ptmat;
}

cv::Mat HeadPose::ptmat2singleColsShape(cv::Mat& ptmat) {
	CV_Assert(ptmat.type() == CV_32FC1 && ptmat.rows == 2);
	int ptnums = ptmat.cols;
	cv::Mat shape(2 * ptnums, 1, CV_32FC1);
	for (int i = 0; i < landnum_; i++) {
		shape.at<float>(2 * i, 0) = ptmat.at<float>(0, i);
		shape.at<float>(2 * i + 1, 0) = ptmat.at<float>(1, i);
	}
	std::cout << "shape size: " << shape.size() << std::endl;
	return shape;
}

void HeadPose::normalizeRows(cv::Mat& src, int rowIndex /* = -1 */) {
	if (rowIndex > src.rows - 1) {
		throw std::exception("row index exceed");
	}
	double minValue, maxValue;
	if (rowIndex == -1) { // normalize all rows
		for (int i = 0; i < src.rows; i++) {
			cv::Mat rowmat = src.row(i);
			cv::minMaxLoc(rowmat, &minValue, &maxValue);
			rowmat = (rowmat - minValue) / (maxValue - minValue);
		}
	}
	else {
		cv::Mat rowmat = src.row(rowIndex);
		cv::minMaxLoc(rowmat, &minValue, &maxValue);
		rowmat = (rowmat - minValue) / (maxValue - minValue);
	}
}

cv::Mat HeadPose::estimate2dRotate(cv::Mat& fromShape, cv::Mat& toShape, double * pangle) {
	AffineTransform atf = customCV::impl::SimilarityTransform(fromShape, toShape);
	if (pangle != NULL) {
		*pangle = asin(atf.getRotation_unscale().at<float>(0, 1));
	}
	cv::Mat tform = atf.getRotation();
	std::cout << "tform: " << tform << std::endl;
	cv::Mat b = atf.getB();
	cv::Mat fromPtmat = singleColShape2ptmat(fromShape);
	cv::Mat rotateMat = tform * fromPtmat;
	for (int col = 0; col < rotateMat.cols; col++) {
		rotateMat.col(col) = rotateMat.col(col) + b;
	}
	return rotateMat;
}

//ptmat is 2 * n matrix
cv::Mat HeadPose::tform2image(cv::Mat& ptmat, cv::Rect rect) {
	AffineTransform tform_to_img = customCV::impl::unnormalizing_tform(rect);
	return tform_to_img(ptmat);
}

cv::Mat HeadPose::image2tform(cv::Mat& ptmat, cv::Rect rect) {
	AffineTransform tform_from_img = customCV::impl::normalizing_tform(rect);
	return tform_from_img(ptmat);
}


void HeadPose::resetAngle() {
	angleX_ = 0;
	angleY_ = 0;
	angleZ_ = 0;

	std::cout << "reset angle " << std::endl;
	std::cout << angleZ_ * 180.0 / CV_PI << " " << angleY_ * 180.0 / CV_PI << " " << angleX_ * 180.0 / CV_PI << std::endl;
}


void HeadPose::visualize() {
	//std::cout << "rotatePtmat" << std::endl << rotatePtmat_ << std::endl;
	std::cout << "angle: " << angleZ_ << std::endl;
	cv::Mat debugView(cv::Size(500, 500), CV_32FC3, cv::Scalar(0, 0, 0));
	showNormalizePtmat(debugView, initShapePtmat_);
	showNormalizePtmat(debugView, detPtMat_, cv::Scalar(255, 0, 255));

	cv::Mat projectView(cv::Size(500, 500), CV_32FC3, cv::Scalar(0, 0, 0));
	showNormalizePtmat(projectView, rotatePtmat_);
	showNormalizePtmat(projectView, detPtMat_, cv::Scalar(255, 0, 255));

	cv::imshow("es", image_);
	//cv::imshow("rotate", projectView);
	//cv::imshow("debugview", debugView);
	cv::waitKey(20);
}



//void  draw(cv::Mat& src) {
//	for (int i = 0; i < _pts.size(); i++) {
//		cv::circle(src, _pts[i], 3, cv::Scalar(255, 0, 255), -1);
//	}
//	cv::rectangle(src, _rect, cv::Scalar(0, 255, 0), 2);
//}


#endif