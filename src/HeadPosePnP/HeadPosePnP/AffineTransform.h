#ifndef AFFINE_TRANSFORM_H_H
#define AFFINE_TRANSFORM_H_H 

#include "common.h"

struct AffineTransform {
	cv::Mat_<float> rotation;
	cv::Mat_<float> b;
	float c;
	AffineTransform(cv::Mat_<float> rotation_, cv::Mat_<float> b_, float c_) {
		rotation_.copyTo(rotation);
		b_.copyTo(b);
		c = c_;
	}
	cv::Mat getRotation() {
		return rotation * c;
	}

	cv::Mat getRotation_unscale() {
		return rotation;
	}
	cv::Mat getB() {
		return b;
	}


	cv::Mat operator()(const cv::Mat& locateMat) {
		cv::Mat ret;
		ret = c * rotation * locateMat;
		for (int i = 0; i < locateMat.cols; i++) {
			ret.at<float>(0, i) = ret.at<float>(0, i) + b.at<float>(0, 0);
		}
		for (int i = 0; i < locateMat.cols; i++) {
			ret.at<float>(1, i) = ret.at<float>(1, i) + b.at<float>(1, 0);
		}
		return ret;
	}
};

cv::Mat pts2Mat(const std::vector<cv::Point2f> pts);

std::vector<cv::Point2f> mat2Pts(const cv::Mat& ptmat);


//calc the affine transform matrix with 2 x n point mat
AffineTransform SimilarityTransform(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2);


AffineTransform SimilarityTransformPts(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

std::vector<cv::Point2f> estimate2dRotate(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, float *pangle = NULL);

std::vector<cv::Point2f> estimate2dRotate(std::vector<cv::Point2f>& queryPts, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, float *pangle = NULL);

float estimate2dRotateAngle(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2);

std::vector<cv::Point2f> estimate2dRotateDelta(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<cv::Point2f> deltaPts);

#endif