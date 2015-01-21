#ifndef ESTIMATE_POS_H 
#define ESTIMATE_POS_H
#include <iostream>
#include <windows.h>
#include <vector>
#include "common.h"
#include "dlib/opencv.h"
#include <opencv2/viz/vizcore.hpp>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
#include "esrShape.h"
#include "view3d.h"

class EstimatePos {
public:
	EstimatePos(const char* facemodel, const char* shapemodel, const char* plyname, const char* indexName, int landnum = 74) {
		pesr_ = new EsrShape(facemodel, shapemodel, landnum);
		pv3d_ = new View3D(plyname, indexName, landnum);
		landnum_ = landnum;
		init();
	}
	EstimatePos(const char* facemodel, const char* shapemodel, int landnum = 74) {
		pesr_ = new EsrShape(facemodel, shapemodel, landnum);
		landnum_ = landnum;
		pv3d_ = NULL;
		init();
	}

	void init();
	~EstimatePos() {
		delete pesr_;
		if (pv3d_ != NULL) {
			delete pv3d_;
		}
	}
	void doEstimatePos(const cv::Mat &src);
	void doEstimatePos3d(const cv::Mat& src);
	void show2dProject(cv::Mat dstView, cv::Mat srcmat);

	//change vector point  or mat point to the uniform format .
	cv::Mat pts2Mat(const std::vector<cv::Point2f> pts);
	cv::Mat muitiChanelMat2singleChannel(cv::Mat muitiMat);
	cv::Mat singleChanelMat2muitiChannel(cv::Mat singleMat);
	cv::Mat estimate2dRotate(cv::Mat& fromPtmat, cv::Mat& toPtmat, double *pangle = NULL);
	void normalizeRows(cv::Mat& src, int rowIndex = -1);
	cv::Mat tform2image(cv::Mat& ptmat);
	cv::Mat image2tform(cv::Mat& ptmat);
	cv::Mat singleColShape2ptmat(cv::Mat oneColMat);
	cv::Mat ptmat2singleColsShape(cv::Mat& ptmat);
	//double estimate2dRotateAngle(cv::Mat fromPtmat, cv::Mat toPtmat);
	void showNormalizePtmat(cv::Mat &view, cv::Mat ptmat, cv::Scalar color = cv::Scalar(255, 255, 255));
	int detect(const cv::Mat& src);
	void visualize();
	//friend class EsrShape;
private:
	float angleX_, angleY_, angleZ_;
	cv::Mat image_;
	EsrShape *pesr_;
	int landnum_;
	View3D *pv3d_;
	cv::Mat initShape_;
	cv::Mat initShapePtmat_;
	cv::Mat detPtMat_;
	cv::Mat tformDetPtmat_;
	cv::Mat rotatePtmat_;
	
};

void EstimatePos::show2dProject(cv::Mat dstView, cv::Mat srcmat) {

}

void EstimatePos::init() {
	if (pv3d_ == NULL) {
		initShape_ = pesr_->getInitShape();
		initShapePtmat_ = singleColShape2ptmat(initShape_);
	}
	else {
		initShapePtmat_ = pv3d_->getFrointfacePtmat();
		initShapePtmat_.row(1) = 1.0 - initShapePtmat_.row(1);
		initShape_ = ptmat2singleColsShape(initShapePtmat_);
	}
}

void EstimatePos::showNormalizePtmat(cv::Mat &view, cv::Mat ptmat, cv::Scalar color) {
	for (int k = 0; k < ptmat.cols; k++) {
		int x = ptmat.at<float>(0, k) * view.cols;
		int y = ptmat.at<float>(1, k) * view.rows;
		cv::circle(view, cv::Point(x, y), 2, color, -1);
	}
}

//row present the corridate . col present the diffent point
cv::Mat EstimatePos::pts2Mat(const std::vector<cv::Point2f> pts) {
	cv::Mat ptmat(2, pts.size(), CV_32FC1);
	for (int i = 0; i < pts.size(); i++) {
		ptmat.at<float>(0, i) = pts[i].x;
		ptmat.at<float>(1, i) = pts[i].y;
	}
	return ptmat;
}

//muitlchannelmat is in the format (1 row, n cols)
//only process CV_32FC3
cv::Mat EstimatePos::muitiChanelMat2singleChannel(cv::Mat muitiMat) {
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

cv::Mat EstimatePos::singleChanelMat2muitiChannel(cv::Mat singleMat) {
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
cv::Mat EstimatePos::singleColShape2ptmat(cv::Mat oneColMat) {
	CV_Assert(oneColMat.rows % 2 == 0 && oneColMat.type() == CV_32FC1);
	int ptnums = oneColMat.rows / 2;
	cv::Mat ptmat(2, ptnums, CV_32FC1);
	for (int k = 0; k < landnum_; k++) {
		ptmat.at<float>(0, k) = oneColMat.at<float>(2 * k, 0);
		ptmat.at<float>(1, k) = oneColMat.at<float>(2 * k + 1, 0);
	}
	return ptmat;
}

cv::Mat EstimatePos::ptmat2singleColsShape(cv::Mat& ptmat) {
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

void EstimatePos::normalizeRows(cv::Mat& src, int rowIndex /* = -1 */) {
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

cv::Mat EstimatePos::estimate2dRotate(cv::Mat& fromShape, cv::Mat& toShape, double * pangle) {
	AffineTransform atf = customCV::impl::SimilarityTransform(fromShape, toShape);
	if (pangle != NULL) {
		*pangle = asin(atf.getRotation_unscale().at<float>(0, 1)) * 180.0 / CV_PI;
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
cv::Mat EstimatePos::tform2image(cv::Mat& ptmat) {
	AffineTransform tform_to_img = customCV::impl::unnormalizing_tform(pesr_->getRect());
	return tform_to_img(ptmat);
}

cv::Mat EstimatePos::image2tform(cv::Mat& ptmat) {
	AffineTransform tform_from_img = customCV::impl::normalizing_tform(pesr_->getRect());
	return tform_from_img(ptmat);
}

void EstimatePos::doEstimatePos3d(const cv::Mat& src) {

}


int EstimatePos::detect(const cv::Mat& src) {
	image_ = src.clone();
	pesr_->detect(image_);
	std::vector<cv::Point2f> pts = pesr_->getPts();
	std::cout << pts.size() << std::endl;
	if (pts.size() <= 0) {
		return 0;
	}
	detPtMat_ = pts2Mat(pts);
	tformDetPtmat_ = image2tform(detPtMat_);
	return 1;
}

void EstimatePos::doEstimatePos(const cv::Mat& src) {
	if (!detect(src)) {
		return;
	}
	double angle;
	
	rotatePtmat_ = estimate2dRotate(initShape_, ptmat2singleColsShape(tformDetPtmat_), &angle);
	angleZ_ = angle;
	visualize();
	
}

void EstimatePos::visualize() {
	cv::Mat debugView(cv::Size(500, 500), CV_32FC3, cv::Scalar(0, 0, 0));
	showNormalizePtmat(debugView, initShapePtmat_);
	std::cout << "rotatePtmat" << std::endl << rotatePtmat_ << std::endl;
	std::cout << "angle: " << angleZ_ << std::endl;
	normalizeRows(rotatePtmat_);
	normalizeRows(detPtMat_);

	cv::Mat projectView(cv::Size(500, 500), CV_32FC3, cv::Scalar(0, 0, 0));
	showNormalizePtmat(projectView, rotatePtmat_);
	showNormalizePtmat(projectView, detPtMat_, cv::Scalar(255, 0, 255));

	pesr_->draw(image_);
	cv::imshow("es", image_);
	cv::imshow("rotate", projectView);
	cv::imshow("debugview", debugView);
	cv::waitKey();
}



#endif