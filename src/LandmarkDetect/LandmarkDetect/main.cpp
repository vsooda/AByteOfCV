//#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
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
#include "estimatePos.h"
#include "view3d.h"

const int landmark_num = 74;

using namespace dlib;
using namespace std;

typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > frontal_face_detector;
cv::Mat doProject(cv::Mat ptMat);



void face_landmark()
{
	try
	{
		frontal_face_detector detector;
		deserialize("frontface.dat") >> detector;
		customCV::shape_predictor sp;
		sp = load_ft<customCV::shape_predictor>("D:/data/74.yaml");

		std::vector<string> names;
		string dir;
		int cnt = readDir("D:/wkdir/helen_3/*.jpg", names, dir);
		for (int i = 0; i < cnt; i++) {
			string filename = dir + names[i];
			cout << "processing image " << filename << endl;
			array2d<rgb_pixel> img;
			cv::Mat src = cv::imread(filename.c_str());
			cv::resize(src, src, cv::Size(500, 500));
			dlib::cv_image<rgb_pixel> *pimg = new dlib::cv_image<rgb_pixel>(src);
			assign_image(img, *pimg);

			//pyramid_up(img);
			cv::Mat src2;
			src.convertTo(src2, CV_32FC3);
			cv::Mat avg(src2.size(), CV_32FC1, cv::Scalar(0));
			for (int x = 0; x < avg.cols; x++) {
				for (int y = 0; y < avg.rows; y++) {
					cv::Vec3f temp = src2.at<cv::Vec3f>(y, x);
					avg.at<float>(y, x) = (temp[0] + temp[1] + temp[2]) / 3;
				}
			}

			std::vector<dlib::rectangle> dets = detector(img);
			if (dets.size() <= 0) {
				continue;
			}

			std::vector<cv::Mat> shapes;  //136 x 1
			std::vector<cv::Mat> currentMats; //landmark_num x 2
			std::vector<cv::Mat> uncurrentMats; //landmark_num x 2
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				cv::Rect rect;
				rect.x = dets[j].left();
				rect.y = dets[j].top();
				rect.width = dets[j].right() - dets[j].left();
				rect.height = dets[j].bottom() - dets[j].top();
				cv::Mat shape = sp(avg, rect);
				AffineTransform tform_to_img = customCV::impl::unnormalizing_tform(rect);
				cv::Mat currentMat(2, landmark_num, CV_32F);
				for (int k = 0; k < landmark_num; k++) {
					currentMat.at<float>(0, k) = shape.at<float>(2 * k, 0);
					currentMat.at<float>(1, k) = shape.at<float>(2 * k + 1, 0);
				}
				uncurrentMats.push_back(currentMat);
				currentMat = tform_to_img(currentMat);
				currentMats.push_back(currentMat);
				shapes.push_back(shape);
			}

			for (int j = 0; j < currentMats.size(); j++) {
				cv::Mat res = currentMats[j];
				for (int k = 0; k < landmark_num; k++) {
					int tempx = res.at<float>(0, k);
					int tempy = res.at<float>(1, k);
					cv::circle(src, cv::Point(tempx, tempy), 2, cv::Scalar(255, 255, 255));
				}
			}
			cv::Rect rect(cv::Point(dets[0].left(), dets[0].top()), cv::Point(dets[0].right(), dets[0].bottom()));
			//for (int k = 0; k < dets.size(); k++) {
				//cv::rectangle(src, cv::Point(dets[k].left(), dets[k].top()),
				//	cv::Point(dets[k].right(), dets[k].bottom()), cv::Scalar(255, 0, 0));
			//}
			cv::rectangle(src, rect, cv::Scalar(255, 0, 0));
			//std::cout << sp.getInitShape() << std::endl;
			cv::Mat initShape = sp.getInitShape();
			cv::Mat initShapeImageMat(2, landmark_num, CV_32FC1);
			for (int k = 0; k < landmark_num; k++) {
				initShapeImageMat.at<float>(0, k) = initShape.at<float>(2 * k, 0);
				initShapeImageMat.at<float>(1, k) = initShape.at<float>(2 * k + 1, 0);
			}
			AffineTransform tform_to_img = customCV::impl::unnormalizing_tform(rect);

			AffineTransform atf = customCV::impl::SimilarityTransform(initShape, shapes[0]);
			//AffineTransform atf = customCV::impl::SimilarityTransform(shapes[0], initShape);
			cv::Mat tform = atf.getRotation();
			cv::Mat b = atf.getB();
			std::cout << "tform" << tform  << " " << atf.getRotation_unscale() << std::endl;
			double angle = asin(atf.getRotation_unscale().at<float>(0, 1)) * 180.0 / CV_PI;
			std::cout << "rotation angle: " << angle << std::endl;
			cv::Mat projectMat = tform * initShapeImageMat;
			for (int col = 0; col < projectMat.cols; col++) {
				projectMat.col(col) = projectMat.col(col) + b;
			}
			
			initShapeImageMat = tform_to_img(initShapeImageMat);
			projectMat = tform_to_img(projectMat);
			
			//norminize..
			cv::Mat pmat(src.size(), CV_8UC3);
			double minValue, maxValue;
			cv::Mat temp;
			for (int row = 0; row < 2; row++) {
				temp = currentMats[0].row(row);
				minMaxLoc(temp, &minValue, &maxValue);
				temp = (temp - minValue) / (maxValue - minValue);
			}
			for (int row = 0; row < 2; row++) {
				temp = projectMat.row(row);
				minMaxLoc(temp, &minValue, &maxValue);
				temp = (temp - minValue) / (maxValue - minValue);
			}

			currentMats[0] = currentMats[0] * 500;
			projectMat = projectMat * 500;

			for (int k = 0; k < landmark_num; k++) {
				int tempx = projectMat.at<float>(0, k);
				int tempy = projectMat.at<float>(1, k);
				int tempx1 = currentMats[0].at<float>(0, k);
				int tempy1 = currentMats[0].at<float>(1, k);
				cv::circle(pmat, cv::Point(tempx, tempy), 3, cv::Scalar(255, 255, 255), -1);
				cv::circle(pmat, cv::Point(tempx1, tempy1), 2, cv::Scalar(255, 0, 255), -1);
			}

			cv::Mat initmat(src.size(), CV_8UC3);
			for (int k = 0; k < landmark_num; k++) {
				int tempx2 = initShapeImageMat.at<float>(0, k);
				int tempy2 = initShapeImageMat.at<float>(1, k);
				cv::circle(initmat, cv::Point(tempx2, tempy2), 5, cv::Scalar(255, 255, 255));
			}
	
			cv::imshow("project shape", pmat);
			cv::imshow("init shape", initmat);
			cv::imshow("dst", src);
			cv::waitKey();
		}

	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}




void face_landmark1()
{
	try
	{
		//frontal_face_detector detector = get_frontal_face_detector();
		frontal_face_detector detector;
		deserialize("frontface.dat") >> detector;
		dlib1::shape_predictor sp;
		//deserialize("D:/data/shape_predictor_68_face_landmarks.dat") >> sp;
		deserialize("sp_10000.dat") >> sp;
		save_ft("D:/data/74.yaml", sp);
		//return;
		//sp = load_ft<dlib1::shape_predictor>("D:/data/74.yaml");
		//save_ft("D:/data/13.yaml", sp);
		//return;

		std::vector<string> names;
		string dir;
		int cnt = readDir("D:/wkdir/face/*.jpg", names, dir);
		for (int i = 0; i < cnt; i++) {
			string filename = dir + names[i];
			cout << "processing image " << filename << endl;
			array2d<rgb_pixel> img;
			cv::Mat src = cv::imread(filename.c_str());
			dlib::cv_image<rgb_pixel> *pimg = new dlib::cv_image<rgb_pixel>(src);
			assign_image(img, *pimg);

			//load_image(img, filename.c_str());
			//pyramid_up(img);
			cv::Mat src2;
			src.convertTo(src2, CV_32FC3);
			cv::Mat avg(src2.size(), CV_32FC1, cv::Scalar(0));
			for (int x = 0; x < avg.cols; x++) {
				for (int y = 0; y < avg.rows; y++) {
					cv::Vec3f temp = src2.at<cv::Vec3f>(y, x);
					avg.at<float>(y, x) = (temp[0] + temp[1] + temp[2]) / 3;
				}
			}


			std::vector<dlib::rectangle> dets = detector(img);

			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(avg, dets[j]);
				shapes.push_back(shape);
			}

			int scale = 1;
			for (int j = 0; j < shapes.size(); j++) {
				full_object_detection res = shapes[j];
				rectangle rect = res.get_rect();
				cv::rectangle(src, cv::Point(rect.left() / scale, rect.top() / scale), cv::Point(rect.right() / scale, rect.bottom() / scale), cv::Scalar(255, 0, 0));
				for (int i = 0; i < landmark_num; i++) {
					point pt = res.part(i);
					int tempx = pt.x() / scale;
					int tempy = pt.y() / scale;
					cv::circle(src, cv::Point(tempx, tempy), 1, cv::Scalar(255, 255, 0));
				}
			}

			cv::imshow("dst", src);
			cv::waitKey();
		}

	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}




double interocular_distance(
	const full_object_detection& det
	)
{
	dlib::vector<double, 2> l, r;
	double cnt = 0;
	// Find the center of the left eye by averaging the points around 
	// the eye.
	for (unsigned long i = 36; i <= 41; ++i)
	{
		l += det.part(i);
		++cnt;
	}
	l /= cnt;

	// Find the center of the right eye by averaging the points around 
	// the eye.
	cnt = 0;
	for (unsigned long i = 42; i <= 47; ++i)
	{
		r += det.part(i);
		++cnt;
	}
	r /= cnt;

	// Now return the distance between the centers of the eyes
	return length(l - r);
}

std::vector<std::vector<double> > get_interocular_distances(
	const std::vector<std::vector<full_object_detection> >& objects
	)
{
	std::vector<std::vector<double> > temp(objects.size());
	for (unsigned long i = 0; i < objects.size(); ++i)
	{
		for (unsigned long j = 0; j < objects[i].size(); ++j)
		{
			temp[i].push_back(interocular_distance(objects[i][j]));
		}
	}
	return temp;
}


cv::Mat cvcloud_load()
{
	cv::Mat cloud(1, 1889, CV_32FC3);
	ifstream ifs("bunny.ply");

	string str;
	for (size_t i = 0; i < 12; ++i)
		getline(ifs, str);

	cv::Point3f* data = cloud.ptr<cv::Point3f>();
	float dummy1, dummy2;
	for (size_t i = 0; i < 1889; ++i)
		ifs >> data[i].x >> data[i].y >> data[i].z >> dummy1 >> dummy2;

	cloud *= 5.0f;
	return cloud;
}




void vizTest_bak() {

	bool camera_pov = true;

	/// Create a window
	cv::viz::Viz3d myWindow("Coordinate Frame");

	/// Add coordinate axes
	myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

	///*cv::viz::WLine axis(cv::Point3f(-1.0f, -1.0f, -1.0f), cv::Point3f(1.0f, 1.0f, 1.0f));
	//axis.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	//myWindow.showWidget("Line Widget", axis);*/

	//cv::viz::WCube cube_widget(cv::Point3f(0.5, 0.5, 0.0), cv::Point3f(0.0, 0.0, -0.5), true, cv::viz::Color::blue());
	//cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);

	///// Display widget (update if already displayed)
	//myWindow.showWidget("Cube Widget", cube_widget);

	//cv::Mat rot_vec = cv::Mat::zeros(1, 3, CV_32F);
	//float translation_phase = 0.0, translation = 0.0;
	//while (!myWindow.wasStopped())
	//{
	//	/* Rotation using rodrigues */
	//	/// Rotate around (1,1,1)
	//	rot_vec.at<float>(0, 0) += CV_PI * 0.01f;
	//	rot_vec.at<float>(0, 1) = 2.0f;
	//	rot_vec.at<float>(0, 2) += 0.0f;
	//	//rot_vec.at<float>(0, 1) += CV_PI * 0.01f;
	//	//rot_vec.at<float>(0, 2) += CV_PI * 0.01f;

	//	/// Shift on (1,1,1)
	//	//translation_phase += CV_PI * 0.01f;
	//	translation_phase = 0;
	//	translation = sin(translation_phase);

	//	cv::Mat rot_mat;
	//	Rodrigues(rot_vec, rot_mat);

	//	/// Construct pose
	//	cv::Affine3f pose(rot_mat, cv::Vec3f(translation, translation, translation));

	//	myWindow.setWidgetPose("Cube Widget", pose);

	//	myWindow.spinOnce(1, true);
	//}


	/// Let's assume camera has the following properties
	//cv::Point3d cam_pos(3.0f, 3.0f, 3.0f), cam_focal_point(3.0f, 3.0f, 2.0f), cam_y_dir(-1.0f, 0.0f, 0.0f);
	cv::Point3d cam_pos(0.0f, 0.0f, 2.0f), cam_focal_point(0.0f, 0.0f, 0.10f), cam_y_dir(0.0f, -1.0f, 0.0f);

	/// We can get the pose of the cam using makeCameraPose
	cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

	/// We can get the transformation matrix from camera coordinate system to global using
	/// - makeTransformToGlobal. We need the axes of the camera
	//cv::Affine3f transform = cv::viz::makeTransformToGlobal(cv::Vec3f(0.0f, -1.0f, 0.0f), cv::Vec3f(-1.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, -1.0f), cam_pos);
	//使物体的坐标发生变化。坐标系本身没有变化
	cv::Affine3f transform = cv::viz::makeTransformToGlobal(cv::Vec3f(-1.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, -1.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, 1.0f), cam_pos);

	/// Create a cloud widget.
	cv::Mat bunny_cloud = cvcloud_load();
	cv::viz::WCloud cloud_widget(bunny_cloud, cv::viz::Color::green());

	/// Pose of the widget in camera frame
	//相对于摄像机的偏移
	cv::Affine3f cloud_pose = cv::Affine3f().translate(cv::Vec3f(0.0f, 0.0f, 0.0f));
	/// Pose of the widget in global frame
	cv::Affine3f cloud_pose_global = transform * cloud_pose;
	//cv::Affine3f cloud_pose_global = cloud_pose;

	/// Visualize camera frame
	if (!camera_pov)
	{
		cv::viz::WCameraPosition cpw(0.5); // Coordinate axes
		cv::viz::WCameraPosition cpw_frustum(cv::Vec2f(0.889484, 0.523599)); // Camera frustum
		myWindow.showWidget("CPW", cpw, cloud_pose);
		myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cloud_pose);
	}

	/// Visualize widget
	myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);

	/// Set the viewer pose to that of camera
	if (camera_pov)
		myWindow.setWidgetPose("bunny", cloud_pose);

	/// Start event loop.
	myWindow.spin();
}

cv::Mat cvcloud_load_base()
{
	cv::Mat cloud(1, 1952, CV_32FC3);
	ifstream ifs("D:/data/base.ply");

	string str;
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

cv::Mat getSelectPtMat(const cv::Mat& fullMat) {
	int index[landmark_num];
	ifstream ifs("D:/data/pt.txt");
	for (int i = 0; i < landmark_num; i++) {
		ifs >> index[i];
		std::cout << index[i] << std::endl;
	}
	ifs.close();
	cv::Mat selectPt(1, landmark_num, CV_32FC3);
	for (int i = 0; i < landmark_num; i++) {
		selectPt.at<cv::Vec3f>(0, i) = fullMat.at<cv::Vec3f>(0, index[i]);
	}
	return selectPt;
}

void showRotate(cv::Mat pts, cv::viz::Viz3d myWindow, cv::Affine3f transform) {
	cv::namedWindow("project", 0);
	while (!myWindow.wasStopped()) {
		cv::Mat rot_vec = cv::Mat::zeros(1, 3, CV_32F);
		int cnt = 0;
		for (int i = 2; i >= 0; i--) {
			cnt = 0;
			while (!myWindow.wasStopped())
			{
				if (cnt++ > 20) {
					break;
				}
				/* Rotation using rodrigues */
				/// Rotate around (1,1,1)
				rot_vec.at<float>(0, i) = CV_PI * 0.005f * cnt;
				//rot_vec.at<float>(0, 1) += CV_PI * 0.01f;
				//rot_vec.at<float>(0, 2) += CV_PI * 0.01f;

				/// Shift on (1,1,1)
				//translation_phase += CV_PI * 0.01f;

				cv::Mat rot_mat;
				Rodrigues(rot_vec, rot_mat);
				cv::Affine3f pose(rot_mat);
				//cv::Vec3f pt = pts.at<cv::Vec3f>(0, 0);
				//std::cout << pt << " to " << pose * pt << std::endl;
				cv::Mat mvpResult;
				pts.copyTo(mvpResult);
				for (int j = 0; j < landmark_num; j++) {
					mvpResult.at<cv::Vec3f>(0, j) = pose * pts.at<cv::Vec3f>(0, j);
				}
				cv::Mat view2d = doProject(mvpResult);
				cv::imshow("project", view2d);
				cv::waitKey(20);
				
				myWindow.setWidgetPose("bunny", transform * pose);
				myWindow.spinOnce(1, true);
			}
		}
	}
}

void vizTest() {

	bool camera_pov = true;
	cv::viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
	cv::Point3d cam_pos(0.0f, 0.0f, 5.0f), cam_focal_point(0.0f, 0.0f, 4.0f), cam_y_dir(0.0f, -1.0f, 0.0f);
	//cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);
	//使物体的坐标发生变化。坐标系本身没有变化
	cv::Affine3f transform = cv::viz::makeTransformToGlobal(cv::Vec3f(-1.0f, 0.0f, 0.0f), cv::Vec3f(0.0f, -1.0f, 0.0f), cv::Vec3f(0.0f, 0.0f, -1.0f), cam_pos);

	cv::Mat raw_cloud = cvcloud_load_base();
	cv::Mat bunny_cloud = getSelectPtMat(raw_cloud);
	//cv::Mat bunny_cloud = raw_cloud;
	/*std::cout << bunny_cloud.size() << std::endl;
	cv::FileStorage fs("pt.yml", cv::FileStorage::WRITE);
	fs << "ftmatrix" << bunny_cloud;
	fs.release();*/

	cv::viz::WCloud cloud_widget(bunny_cloud, cv::viz::Color::green());

	cv::Affine3f cloud_pose = cv::Affine3f().translate(cv::Vec3f(0.0f, 0.0f, 0.0f));
	cv::Affine3f cloud_pose_global = transform * cloud_pose;

	if (!camera_pov)
	{
		cv::viz::WCameraPosition cpw(0.5); // Coordinate axes
		cv::viz::WCameraPosition cpw_frustum(cv::Vec2f(0.889484, 0.523599)); // Camera frustum
		myWindow.showWidget("CPW", cpw, cloud_pose);
		myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cloud_pose);
	}

	myWindow.showWidget("bunny", cloud_widget, cloud_pose_global);
	if (!camera_pov) {
		showRotate(bunny_cloud, myWindow, transform);
	}
	else {
		showRotate(bunny_cloud, myWindow, cv::Affine3f());
	}


	myWindow.spin();
}

cv::Mat doProject(cv::Mat ptMat) {
//	std::cout << ptMat << std::endl;
	std::vector<cv::Point> pts;
	cv::Mat channelMat(3, ptMat.cols,  CV_32FC1);
//	std::cout << channelMat.size() << std::endl;
	for (int i = 0; i < ptMat.cols; i++) {
		cv::Vec3f temp = ptMat.at<cv::Vec3f>(0, i);
		channelMat.at<float>(0, i) = temp[0];
		channelMat.at<float>(1, i) = temp[1];
		channelMat.at<float>(2, i) = temp[2];
	}
	

	cv::Mat projectMat(2, ptMat.cols, CV_32FC1);
	for (int i = 0; i < projectMat.cols; i++) {
		for (int j = 0; j < projectMat.rows; j++) {
//			std::cout << channelMat.at<float>(j, i) << " " << channelMat.at<float>(2, i) << "  " << channelMat.at<float>(j, i) / channelMat.at<float>(2, i) << std::endl;
			projectMat.at<float>(j, i) = channelMat.at<float>(j, i)  / (5.0 - channelMat.at<float>(2, i));
		}
	}

	double minValue, maxValue;
	for (int row = 0; row < 3; row++) {
		cv::Mat rowmat = channelMat.row(row);
		cv::minMaxLoc(rowmat, &minValue, &maxValue);
//		std::cout << minValue << " " << maxValue << std::endl;
		rowmat = (rowmat - minValue) / (maxValue - minValue);
	}
	
//	std::cout << "projectMat " << projectMat << std::endl;

	for (int row = 0; row < 2; row++) {
		cv::Mat rowmat = projectMat.row(row);
		cv::minMaxLoc(rowmat, &minValue, &maxValue);
//		std::cout << minValue << " " << maxValue << std::endl;
		rowmat = (rowmat - minValue) / (maxValue - minValue);
	}
//	std::cout << "projectMat " << projectMat << std::endl;

	cv::Mat showView(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
	cv::Mat projectView(cv::Size(500, 500), CV_8UC3, cv::Scalar(0, 0, 0));
	for (int i = 0; i < channelMat.cols; i++) {
			//cv::Point pt(channelMat.at<float>(0, i) * 500, (1.0 - channelMat.at<float>(1, i)) * 500);
			//cv::circle(showView, pt, 2, cv::Scalar(255, 0, 0), -1);
			cv::Point pt1(projectMat.at<float>(0, i) * 500, (1.0 - projectMat.at<float>(1, i)) * 500);
			cv::circle(showView, pt1, 2, cv::Scalar(255, 255, 255), -1);
	}
	//cv::imshow("show", showView);
	//cv::imshow("project", projectView);
	//cv::waitKey();
	return showView;
	//先分别对xyz做归一化
}

void projectTest() {
	cv::FileStorage fs("D:/data/pt.yml", cv::FileStorage::READ);
	cv::Mat ptMat;
	fs["ftmatrix"] >> ptMat;
	doProject(ptMat);
}


void landmark_test() {
	EsrShape esp("frontface.dat", "D:/data/74.yaml");
	cv::Mat img = cv::imread("D:/data/2.jpg");

	esp.detect(img);
	//std::vector<cv::Point2f> pts = esp.getFilterPts();
	//for (int i = 0; i < pts.size(); i++) {
	//	cv::circle(img, pts[i], 3, cv::Scalar(0, 0, 255), -1);
	//}
	esp.draw(img);
	cv::imshow("dst", img);
	cv::waitKey();
}

void poseEstimateTest() {
	EstimatePos ep("frontface.dat", "D:/data/74.yaml");
	std::vector<string> names;
	string dir;
	int cnt = readDir("D:/wkdir/helen_3/*.jpg", names, dir);
	for (int i = 0; i < cnt; i++) {
		string filename = dir + names[i];
		cout << "processing image " << filename << endl;
		cv::Mat src = cv::imread(filename.c_str());
		cv::resize(src, src, cv::Size(500, 500));
		ep.doEstimatePos(src);
	}
}

void view3dTest() {
	View3D v3d("D:/data/base.ply", "D:/data/pt.txt");
	v3d.showRotate();
}



int main() { 
	view3dTest();
	//poseEstimateTest();
	//return 0;
	//landmark_test();
	//return 0;
	//freopen("cv.txt", "w", stdout);
	//projectTest();
	//vizTest();
	//face_landmark();
	//face_landmark1();
}

