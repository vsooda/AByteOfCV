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
	View3D(cv::Mat rawClound, int landmarks = 74) :
			objWindow_("3d pose"),
			landmark_(landmarks),
			camPosition_(0.0f, 0.0f, 5.0f),
			camFocalPoint_(0.0f, 0.0f, 4.0f),
			camYdir_(0.0f, -1.0f, 0.0f)
	{
		objWindow_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
		transform_ = cv::viz::makeTransformToGlobal(cv::Vec3f(-1.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, -1.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, -1.0f), camPosition_);
		cv::viz::WCloud cloud_widget(rawClound, cv::viz::Color::green());
		cv::Affine3f cloud_pose = cv::Affine3f().translate(cv::Vec3f(0.0f, 0.0f, 0.0f));
		cv::Affine3f cloud_pose_global = transform_ * cloud_pose;
		objWindow_.showWidget("bunny", cloud_widget, cloud_pose_global);
		cv::namedWindow("project", 0);
		angleX_ = 0;
		angleY_ = 0;
		angleZ_ = 0;
		
		std::cout << "v3d init ok" << std::endl;
	}

	~View3D() {

	}
	
	void update(cv::Affine3d pose) {
		objWindow_.setWidgetPose("bunny", pose);
		objWindow_.spinOnce(1, true);
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
	
	cv::Point3d camPosition_, camFocalPoint_, camYdir_;
	cv::Affine3f transform_;
	
	const int landmark_;
	
};
#endif