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
#include "HeadPose.h"

class EstimatePos {
public:
	EstimatePos(const char* facemodel, const char* shapemodel, const char* plyname, const char* indexName, int landnum = 74) {
		pesr_ = new EsrShape(facemodel, shapemodel, landnum);
		pv3d_ = new View3D(plyname, indexName, landnum);
		php_ = new HeadPose(plyname, indexName, landnum);
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
	bool doEstimatePos(const cv::Mat &src);
	bool doEstimatePos3d(const cv::Mat& src);
	
	bool detect(const cv::Mat& src);
	float getAngleZ();
	//friend class EsrShape;
private:
	cv::Mat image_;
	EsrShape *pesr_;
	HeadPose *php_;
	int landnum_;
	View3D *pv3d_;
};


void EstimatePos::init() {
	if (pv3d_ == NULL) {
		php_->initShape_ = pesr_->getInitShape();
		php_->initShapePtmat_ = php_->singleColShape2ptmat(php_->initShape_);
	}
	else {
		php_->initShapePtmat_ = pv3d_->getFrointfacePtmat();
		php_->initShapePtmat_.row(1) = 1.0 - php_->initShapePtmat_.row(1);
		php_->initShape_ = php_->ptmat2singleColsShape(php_->initShapePtmat_);
	}
}

bool EstimatePos::detect(const cv::Mat& src) {
	image_ = src.clone();
	php_->image_ = src.clone();
	if (!pesr_->detect(src)) {
		return false;
	}

	php_->pts_ = pesr_->getPts();
	php_->rect_ = pesr_->getRect();
	php_->setInitMat();
	
	return true;
}

bool EstimatePos::doEstimatePos(const cv::Mat& src) {
	if (!detect(src)) {
		return false;
	}
	double angle;
	
	php_->rotatePtmat_ = php_->estimate2dRotate(php_->initShape_, php_->ptmat2singleColsShape(php_->tformDetPtmat_), &angle);
	php_->angleZ_ = angle;
	php_->normalizeRows(php_->rotatePtmat_);
	php_->normalizeRows(php_->detPtMat_);
	std::cout << php_->angleZ_ * 180.0 / CV_PI << std::endl;
//	cv::imshow("src", src);
	return true;
}

bool EstimatePos::doEstimatePos3d(const cv::Mat& src) {
	php_->resetAngle();
	if (!doEstimatePos(src)) {
		return false;
	}
	php_->visualize();
	pv3d_->searchBestAngle(php_->angleZ_, php_->angleX_, php_->angleY_, php_->detPtMat_);
	std::cout << "estimate result: " << php_->angleZ_ * 180.0 / CV_PI << " " << php_->angleY_ * 180.0 / CV_PI << " " << php_->angleX_ * 180.0 / CV_PI << std::endl;
	return true;
}

float EstimatePos::getAngleZ() {
	return php_->angleZ_;
}

#endif