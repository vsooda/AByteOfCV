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
	HeadPose(const char* plyname, const char* indexName, int landnum = 74)
		: plyname_(plyname),
		selectIndex_(indexName) {
		landnum_ = landnum;
		rawClound_ = readPlyData(plyname_);
		selectClound_ = selectPlyData(rawClound_, selectIndex_);
		setFrontfacePtmat();
		initAxisIndex();
		isSelectPtsError_ = false;
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


	//3d compute
	cv::Mat readPlyData(const char* filename);
	cv::Mat selectPlyData(const cv::Mat& fullmat, const char* selectIndex);
	void doProject(cv::Mat ptMat, cv::Mat& projectMat);
	void searchBestAngle(const float &anglez, float & anglex, float &angley, const cv::Mat& detPtmat);
	void searchSeperate();
	void searchTogether();
	float computeAllError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat);
	float computeYError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat);
	float computeXError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat);
	void setYaxisIndex(std::vector<int> indexs);
	void setXaxisIndex(std::vector<int> indexs);
	void searchYaxis();
	void searchXaxis();
	void setCurrentPtmat(float anglex, float angley, float anglez);
	cv::Affine3f computePose(float anglex, float angley, float anglez);
	void renderProject();
	void renderAndSet2dPtmat();
	void initAxisIndex();

	void setFrontfacePtmat();
	cv::Mat getFrointfacePtmat();


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
	cv::Mat rawClound_;
	cv::Mat selectClound_;
	const char* plyname_;
	const char* selectIndex_;
	cv::Mat ptmat2d_;
	cv::Mat detMat_;
	cv::Mat detMatInverse_;
	cv::Mat frontfacePtmat_;
	cv::Mat view2d_;
	std::vector<int> yAxisIndexs_;
	std::vector<int> xAxisIndexs_;
	bool isSelectPtsError_;
	cv::Affine3f pose_;
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


cv::Mat HeadPose::readPlyData(const char* filename) {
	cv::Mat cloud(1, 1952, CV_32FC3);
	std::ifstream ifs(filename);
	std::string str;
	for (size_t i = 0; i < 14; ++i)
		getline(ifs, str);

	cv::Point3f* data = cloud.ptr<cv::Point3f>();
	int temp1, temp2, temp3, temp4;
	for (size_t i = 0; i < 1952; ++i) {
		ifs >> data[i].x >> data[i].y >> data[i].z >> temp1 >> temp2 >> temp3 >> temp4;
		data[i].y = data[i].y - 160.0;
	}
	ifs.close();
	cloud /= 10.0f;
	return cloud;
}

cv::Mat HeadPose::selectPlyData(const cv::Mat& fullmat, const char* selectIndex) {
	int *index = new int[landnum_];
	std::ifstream ifs(selectIndex);
	for (int i = 0; i < landnum_; i++) {
		ifs >> index[i];
	}
	ifs.close();
	cv::Mat selectPt(1, landnum_, CV_32FC3);
	for (int i = 0; i < landnum_; i++) {
		selectPt.at<cv::Vec3f>(0, i) = fullmat.at<cv::Vec3f>(0, index[i]);
	}
	delete[] index;
	return selectPt;
}


//ptMat: 3d coodate
//projectMat: 2d coodate 
//return: a view of the 2d projection
void HeadPose::doProject(cv::Mat ptMat, cv::Mat& projectMat) {
	std::vector<cv::Point> pts;
	cv::Mat channelMat(3, ptMat.cols, CV_32FC1);
	for (int i = 0; i < ptMat.cols; i++) {
		cv::Vec3f temp = ptMat.at<cv::Vec3f>(0, i);
		channelMat.at<float>(0, i) = temp[0];
		channelMat.at<float>(1, i) = temp[1];
		channelMat.at<float>(2, i) = temp[2];
	}
	projectMat = cv::Mat(2, ptMat.cols, CV_32FC1);
	for (int i = 0; i < projectMat.cols; i++) {
		for (int j = 0; j < projectMat.rows; j++) {
			projectMat.at<float>(j, i) = channelMat.at<float>(j, i) / (5.0 - channelMat.at<float>(2, i));
		}
	}
	double minValue, maxValue;
	for (int row = 0; row < 3; row++) {
		cv::Mat rowmat = channelMat.row(row);
		cv::minMaxLoc(rowmat, &minValue, &maxValue);
		rowmat = (rowmat - minValue) / (maxValue - minValue);
	}

	for (int row = 0; row < 2; row++) {
		cv::Mat rowmat = projectMat.row(row);
		cv::minMaxLoc(rowmat, &minValue, &maxValue);
		rowmat = (rowmat - minValue) / (maxValue - minValue);
	}
}

//attention!!! z y x turns
//angleZ_ is const
void HeadPose::searchBestAngle(const float &anglez, float & anglex, float &angley, const cv::Mat& detPtmat) {
	angleZ_ = anglez;
	angleX_ = anglex;
	angleY_ = angley;
	detPtmat.copyTo(detMat_);
	detPtmat.copyTo(detMatInverse_);
	detMatInverse_.row(1) = 1.0 - detMatInverse_.row(1);
	searchSeperate();
	//searchTogether();
	angley = angleY_;
	anglex = angleX_;
}

void HeadPose::searchSeperate() {
	searchYaxis();
	searchXaxis();
}

void HeadPose::searchTogether() {
	float anglex = -0.5;
	float angley = -0.5;
	float minError = 10000000;
	float delta = 0.03;
	while (anglex < 0.5) {
		angley = -0.5;
		while (angley < 0.5) {
			setCurrentPtmat(anglex, angley, angleZ_);
			//float yerror = computeYError(ptmat2d_, detMatInverse_);
			float allError = computeAllError(ptmat2d_, detMatInverse_);
			if (allError <= minError) {
				minError = allError;
				angleX_ = anglex;
				angleY_ = angley;
				//update();
			}
			angley = angley + delta;
		}
		anglex = anglex + delta;
	}
}

float HeadPose::computeAllError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
	if (!isSelectPtsError_) {
		cv::Mat diffMat = currentPtmat - detPtmat;
		diffMat = diffMat.mul(diffMat);
		cv::Scalar sumError = cv::sum(diffMat);
		return sumError.val[0];
	}
	else {
		return computeXError(currentPtmat, detPtmat) + computeYError(currentPtmat, detPtmat);
	}
}

//only count x error
float HeadPose::computeYError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
	if (!isSelectPtsError_) {
		cv::Mat diffMat = currentPtmat.row(0) - detPtmat.row(0);
		diffMat = diffMat.mul(diffMat);
		cv::Scalar sumError = cv::sum(diffMat);
		//std::cout << diffMat.size() << " sumError: " << sumError << std::endl;
		return sumError.val[0];
	}
	else {
		cv::Mat diffMat = currentPtmat.row(0) - detPtmat.row(0);
		float sumError = 0;
		for (int i = 0; i < yAxisIndexs_.size(); i++) {
			sumError += diffMat.at<float>(0, yAxisIndexs_[i]) * diffMat.at<float>(0, yAxisIndexs_[i]);
		}
		return sumError;
	}
}

//only count y error 
//attention: for the y, must inverse the axise
float HeadPose::computeXError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
	if (!isSelectPtsError_) {
		cv::Mat diffMat = currentPtmat.row(1) - detPtmat.row(1);
		diffMat = diffMat.mul(diffMat);
		cv::Scalar sumError = cv::sum(diffMat);
		//std::cout << diffMat.size() << " sumError: " << sumError << std::endl;
		return sumError.val[0];
	}
	else {
		cv::Mat diffMat = currentPtmat.row(1) - detPtmat.row(1);
		float sumError = 0;
		for (int i = 0; i < xAxisIndexs_.size(); i++) {
			sumError += diffMat.at<float>(0, xAxisIndexs_[i]) * diffMat.at<float>(0, xAxisIndexs_[i]);
		}
		return sumError;
	}
}

void HeadPose::setYaxisIndex(std::vector<int> indexs) {
	yAxisIndexs_.clear();
	for (int i = 0; i < indexs.size(); i++) {
		yAxisIndexs_.push_back(indexs[i]);
	}
}

void HeadPose::setXaxisIndex(std::vector<int> indexs) {
	xAxisIndexs_.clear();
	for (int i = 0; i < indexs.size(); i++) {
		xAxisIndexs_.push_back(indexs[i]);
	}
}



void HeadPose::searchYaxis() {
	float angle = -0.5;
	float minError = 10000000;
	float delta = 0.01;
	while (angle < 0.5) {
		setCurrentPtmat(angleX_, angle, angleZ_);
		float yerror = computeYError(ptmat2d_, detMatInverse_);
		if (yerror <= minError) {
			minError = yerror;
			angleY_ = angle;
			//update();
		}
		angle = angle + delta;
	}
}

void HeadPose::searchXaxis() {
	float angle = -0.5;
	float minError = 10000000;
	float delta = 0.01;
	while (angle < 0.5) {
		setCurrentPtmat(angle, angleY_, angleZ_);
		float xerror = computeXError(ptmat2d_, detMatInverse_);
		if (xerror <= minError) {
			minError = xerror;
			angleX_ = angle;
			//update();
		}
		angle = angle + delta;
	}
}

void HeadPose::setCurrentPtmat(float anglex, float angley, float anglez) {
	//std::cout << "current pose" <<  anglez * 180.0 / CV_PI << " " << angley * 180.0 / CV_PI << " " << anglex * 180.0 / CV_PI << std::endl;
	computePose(anglex, angley, anglez);
	renderAndSet2dPtmat();
}

cv::Affine3f HeadPose::computePose(float anglex, float angley, float anglez) {
	cv::Mat rot_vec = cv::Mat::zeros(1, 3, CV_32F);
	rot_vec.at<float>(0, 0) = anglex;
	rot_vec.at<float>(0, 1) = angley;
	rot_vec.at<float>(0, 2) = anglez;
	cv::Mat rot_mat;
	Rodrigues(rot_vec, rot_mat);
	pose_ =  cv::Affine3f(rot_mat);
	return pose_;
}

void HeadPose::renderProject() {
	view2d_ = cv::Mat(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < ptmat2d_.cols; i++) {
		cv::Point pt1(ptmat2d_.at<float>(0, i) * 500, (1.0 - ptmat2d_.at<float>(1, i)) * 500);
		cv::circle(view2d_, pt1, 2, cv::Scalar(255, 255, 255), -1);
		cv::Point pt(detMat_.at<float>(0, i) * 500, detMat_.at<float>(1, i) * 500);
		cv::circle(view2d_, pt, 2, cv::Scalar(255, 0, 255), -1);
	}
}


void HeadPose::renderAndSet2dPtmat() {
	cv::Mat mvpResult;
	selectClound_.copyTo(mvpResult);
	for (int j = 0; j < landnum_; j++) {
		mvpResult.at<cv::Vec3f>(0, j) = pose_ * selectClound_.at<cv::Vec3f>(0, j);
	}
	doProject(mvpResult, ptmat2d_);
	renderProject();
}

void HeadPose::initAxisIndex() {
	int yIndexs[] = { 7, 55, 59, 64, 62, 49, 39, 65, 29, 30, 32, 34 };
	yAxisIndexs_.assign(yIndexs, yIndexs + 12);
	int xIndexs[] = { 18, 24, 29, 35, 43, 33, 65, 39, 64, 6, 7, 8 };
	xAxisIndexs_.assign(xIndexs, xIndexs + 12);
	/*int indexs[] = {64, 39, 29, 35, 43, 33};
	yAxisIndexs_.assign(indexs, indexs + 6);
	xAxisIndexs_.assign(indexs, indexs + 6);*/
}


void HeadPose::setFrontfacePtmat() {
	doProject(selectClound_, frontfacePtmat_);
	//cv::imshow("frontface", view);
}

cv::Mat HeadPose::getFrointfacePtmat() {
	return frontfacePtmat_;
}



#endif