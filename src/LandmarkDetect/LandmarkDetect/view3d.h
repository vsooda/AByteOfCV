#ifndef VIEW_IN_3D
#define VIEW_IN_3D
#include <iostream>
#include <windows.h>
#include <vector>
#include "common.h"
#include <opencv2/viz/vizcore.hpp>
#include <fstream>
#include <opencv2/calib3d/calib3d.hpp>
#include <cmath>
class View3D {
public:
	View3D(const char* plyname, const char* indexName, int landmarks = 74) :
			objWindow_("3d pose"),
			plyname_(plyname),
			selectIndex_(indexName),
			landmark_(landmarks),
			camPosition_(0.0f, 0.0f, 5.0f),
			camFocalPoint_(0.0f, 0.0f, 4.0f),
			camYdir_(0.0f, -1.0f, 0.0f)
	{
		objWindow_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
		transform_ = cv::viz::makeTransformToGlobal(cv::Vec3f(-1.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, -1.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, -1.0f), camPosition_);
		rawClound_ = readPlyData(plyname_);
		selectClound_ = selectPlyData(rawClound_, selectIndex_);
		//cv::viz::WCloud cloud_widget(selectClound_, cv::viz::Color::green());
		cv::viz::WCloud cloud_widget(rawClound_, cv::viz::Color::green());
		cv::Affine3f cloud_pose = cv::Affine3f().translate(cv::Vec3f(0.0f, 0.0f, 0.0f));
		cv::Affine3f cloud_pose_global = transform_ * cloud_pose;
		objWindow_.showWidget("bunny", cloud_widget, cloud_pose_global);
		cv::namedWindow("project", 0);
		angleX_ = 0;
		angleY_ = 0;
		angleZ_ = 0;
		setFrontfacePtmat();
		initAxisIndex();
		isSelectPtsError_ = false;
		std::cout << "v3d init ok" << std::endl;
	}

	~View3D() {

	}

	cv::Mat readPlyData(const char* filename) {
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

	cv::Mat selectPlyData(const cv::Mat& fullmat, const char* selectIndex) {
		int *index = new int[landmark_];
		std::ifstream ifs(selectIndex);
		for (int i = 0; i < landmark_; i++) {
			ifs >> index[i];
		}
		ifs.close();
		cv::Mat selectPt(1, landmark_, CV_32FC3);
		for (int i = 0; i < landmark_; i++) {
			selectPt.at<cv::Vec3f>(0, i) = fullmat.at<cv::Vec3f>(0, index[i]);
		}
		delete [] index;
		return selectPt;
	}

	void initAxisIndex() {
		int yIndexs[] = { 7, 55, 59, 64, 62, 49, 39, 65, 29, 30, 32, 34};
		yAxisIndexs_.assign(yIndexs, yIndexs + 12);
		int xIndexs[] = { 18, 24, 29, 35, 43, 33, 65, 39, 64, 6, 7, 8 };
		xAxisIndexs_.assign(xIndexs, xIndexs + 12);
		/*int indexs[] = {64, 39, 29, 35, 43, 33};
		yAxisIndexs_.assign(indexs, indexs + 6);
		xAxisIndexs_.assign(indexs, indexs + 6);*/
	}
	
	//ptMat: 3d coodate
	//projectMat: 2d coodate 
	//return: a view of the 2d projection
	void doProject(cv::Mat ptMat, cv::Mat& projectMat) {
		std::vector<cv::Point> pts;
		cv::Mat channelMat(3, ptMat.cols, CV_32FC1);
		for (int i = 0; i < ptMat.cols; i++) {
			cv::Vec3f temp = ptMat.at<cv::Vec3f>(0, i);
			channelMat.at<float>(0, i) = temp[0];
			channelMat.at<float>(1, i) = temp[1];
			channelMat.at<float>(2, i) = temp[2];
		}
		projectMat =  cv::Mat(2, ptMat.cols, CV_32FC1);
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
	void searchBestAngle(const float &anglez, float & anglex, float &angley, const cv::Mat& detPtmat) {
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
		update(angleX_, angleY_, angleZ_);
	}

	void searchSeperate() {
		searchYaxis();
		searchXaxis();
	}

	void searchTogether() {
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
					update();
				}
				angley = angley + delta;
			}
			anglex = anglex + delta;
		}
	}

	float computeAllError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
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
	float computeYError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
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
	float computeXError(const cv::Mat& currentPtmat, const cv::Mat& detPtmat) {
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

	void setYaxisIndex(std::vector<int> indexs) {
		yAxisIndexs_.clear();
		for (int i = 0; i < indexs.size(); i++) {
			yAxisIndexs_.push_back(indexs[i]);
		}
	}

	void setXaxisIndex(std::vector<int> indexs) {
		xAxisIndexs_.clear();
		for (int i = 0; i < indexs.size(); i++) {
			xAxisIndexs_.push_back(indexs[i]);
		}
	}

	

	void searchYaxis() {
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

	void searchXaxis() {
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

	void setCurrentPtmat(float anglex, float angley, float anglez) {
		//std::cout << "current pose" <<  anglez * 180.0 / CV_PI << " " << angley * 180.0 / CV_PI << " " << anglex * 180.0 / CV_PI << std::endl;
		computePose(anglex, angley, anglez);
		renderAndSet2dPtmat();
	}

	void computePose(float anglex, float angley, float anglez) {
		cv::Mat rot_vec = cv::Mat::zeros(1, 3, CV_32F);
		rot_vec.at<float>(0, 0) = anglex;
		rot_vec.at<float>(0, 1) = angley;
		rot_vec.at<float>(0, 2) = anglez;
		cv::Mat rot_mat;
		Rodrigues(rot_vec, rot_mat);
		pose_ = cv::Affine3f(rot_mat);
	}

	void renderProject() {
		view2d_ = cv::Mat(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
		for (int i = 0; i < ptmat2d_.cols; i++) {
			cv::Point pt1(ptmat2d_.at<float>(0, i) * 500, (1.0 - ptmat2d_.at<float>(1, i)) * 500);
			cv::circle(view2d_, pt1, 2, cv::Scalar(255, 255, 255), -1);
			cv::Point pt(detMat_.at<float>(0, i) * 500, detMat_.at<float>(1, i) * 500);
			cv::circle(view2d_, pt, 2, cv::Scalar(255, 0, 255), -1);
		}
	}

	
	void renderAndSet2dPtmat() {
		cv::Mat mvpResult;
		selectClound_.copyTo(mvpResult);
		for (int j = 0; j < landmark_; j++) {
			mvpResult.at<cv::Vec3f>(0, j) = pose_ * selectClound_.at<cv::Vec3f>(0, j);
		}
		doProject(mvpResult, ptmat2d_);
		renderProject();
	}

	void update() {
		cv::imshow("project", view2d_);
		//cv::waitKey(10);
		objWindow_.setWidgetPose("bunny", pose_);
		objWindow_.spinOnce(1, true);
	}

	void update(float anglex, float angley, float anglez) {
		computePose(anglex, angley, anglez);
		renderAndSet2dPtmat();
		update();
	}


	void test() {
		angleX_ = 0.5;
		computePose(angleX_, angleY_, angleZ_);
		renderAndSet2dPtmat();
		update();
		cv::waitKey();
	}

	void showRotate() {
		while (!objWindow_.wasStopped()) {
			int cnt = 0;
			for (int i = 2; i >= 0; i--) {
				cnt = 0;
				while (!objWindow_.wasStopped())
				{
					if (cnt++ > 20) {
						break;
					}
					angleX_ = CV_PI * 0.005f * cnt;
					computePose(angleX_, angleY_, angleZ_);
					renderAndSet2dPtmat();
					update();
				}
			}
		}
	}

	void setFrontfacePtmat() {
		doProject(selectClound_, frontfacePtmat_);
		//cv::imshow("frontface", view);
	}

	cv::Mat getFrointfacePtmat() {
		return frontfacePtmat_;
	}

	void setAngleX(float angle) {
		angleX_ = angle;
	}
	float getAngleX() {
		return angleX_;
	}
	void setAngleY(float angle) {
		angleY_ = angle;
	}
	float getAngleY() {
		return angleY_;
	}
	void setAngleZ(float angle) {
		angleZ_ = angle;
	}
	float getAngleZ() {
		return angleZ_;
	}


private:
	cv::viz::Viz3d objWindow_;
	float angleX_, angleY_, angleZ_;
	cv::Affine3f pose_;
	cv::Point3d camPosition_, camFocalPoint_, camYdir_;
	cv::Affine3f transform_;
	cv::Mat rawClound_;
	cv::Mat selectClound_;
	const char* plyname_;
	const char* selectIndex_;
	const int landmark_;
	cv::Mat ptmat2d_;
	cv::Mat detMat_;
	cv::Mat detMatInverse_;
	cv::Mat frontfacePtmat_;
	cv::Mat view2d_;
	std::vector<int> yAxisIndexs_;
	std::vector<int> xAxisIndexs_;
	bool isSelectPtsError_;
};
#endif