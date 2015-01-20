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
		cv::viz::WCloud cloud_widget(selectClound_, cv::viz::Color::green());
		cv::Affine3f cloud_pose = cv::Affine3f().translate(cv::Vec3f(0.0f, 0.0f, 0.0f));
		cv::Affine3f cloud_pose_global = transform_ * cloud_pose;
		objWindow_.showWidget("bunny", cloud_widget, cloud_pose_global);
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
			std::cout << index[i] << std::endl;
		}
		ifs.close();
		cv::Mat selectPt(1, landmark_, CV_32FC3);
		for (int i = 0; i < landmark_; i++) {
			selectPt.at<cv::Vec3f>(0, i) = fullmat.at<cv::Vec3f>(0, index[i]);
		}
		delete [] index;
		return selectPt;
	}

	cv::Mat doProject(cv::Mat ptMat) {
		std::vector<cv::Point> pts;
		cv::Mat channelMat(3, ptMat.cols, CV_32FC1);
		for (int i = 0; i < ptMat.cols; i++) {
			cv::Vec3f temp = ptMat.at<cv::Vec3f>(0, i);
			channelMat.at<float>(0, i) = temp[0];
			channelMat.at<float>(1, i) = temp[1];
			channelMat.at<float>(2, i) = temp[2];
		}
		cv::Mat projectMat(2, ptMat.cols, CV_32FC1);
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

		cv::Mat showView(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
		for (int i = 0; i < channelMat.cols; i++) {
			cv::Point pt1(projectMat.at<float>(0, i) * 500, (1.0 - projectMat.at<float>(1, i)) * 500);
			cv::circle(showView, pt1, 2, cv::Scalar(255, 255, 255), -1);
		}

		return showView;
	}

	void showRotate() {
		cv::namedWindow("project", 0);
		while (!objWindow_.wasStopped()) {
			cv::Mat rot_vec = cv::Mat::zeros(1, 3, CV_32F);
			int cnt = 0;
			for (int i = 2; i >= 0; i--) {
				cnt = 0;
				while (!objWindow_.wasStopped())
				{
					if (cnt++ > 20) {
						break;
					}
					rot_vec.at<float>(0, i) = CV_PI * 0.005f * cnt;
					cv::Mat rot_mat;
					Rodrigues(rot_vec, rot_mat);
					cv::Affine3f pose(rot_mat);
					cv::Mat mvpResult;
					selectClound_.copyTo(mvpResult);
					for (int j = 0; j < landmark_; j++) {
						mvpResult.at<cv::Vec3f>(0, j) = pose * selectClound_.at<cv::Vec3f>(0, j);
					}
					std::cout << mvpResult << std::endl;
					cv::Mat view2d = doProject(mvpResult);
					cv::imshow("project", view2d);
					cv::waitKey(20);

					objWindow_.setWidgetPose("bunny", pose);
					objWindow_.spinOnce(1, true);
				}
			}
		}
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
};
#endif