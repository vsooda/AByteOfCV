//#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing.h"
#include <iostream>
#include <windows.h>
#include <vector>
#include "common.h"
#include "dlib/opencv.h"

using namespace dlib;
using namespace std;

typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > frontal_face_detector;

int readDir(string path, std::vector<string> &names, string& dir) {
	//vector<string> names;
	dir = path.substr(0, path.find_last_of("\\/") + 1);
	names.clear();
	names.reserve(10000);
	WIN32_FIND_DATAA fileFindData;
	HANDLE hFind = ::FindFirstFileA(path.c_str(), &fileFindData);
	if (hFind == INVALID_HANDLE_VALUE) {
		return 0;
	}

	do{
		if (fileFindData.cFileName[0] == '.')
			continue; // filter the '..' and '.' in the path
		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			continue; // Ignore sub-folders
		names.push_back(fileFindData.cFileName);
	} while (::FindNextFileA(hFind, &fileFindData));
	FindClose(hFind);
	return (int)names.size();
}


// ----------------------------------------------------------------------------------------

template <class T>
T load_ft(const char* fname){
	T x; cv::FileStorage f(fname, cv::FileStorage::READ);
	f["ft object"] >> x; f.release(); return x;
}
//==============================================================================
template<class T>
void save_ft(const char* fname, const T& x){
	cv::FileStorage f(fname, cv::FileStorage::WRITE);
	f << "ft object" << x; f.release();
}

void face_landmark()
{
	try
	{
		frontal_face_detector detector;
		deserialize("frontface.dat") >> detector;
		customCV::shape_predictor sp;
		sp = load_ft<customCV::shape_predictor>("D:/data/1.yaml");

		std::vector<string> names;
		string dir;
		int cnt = readDir("D:/wkdir/images/*.jpg", names, dir);
		for (int i = 0; i < cnt; i++) {
			string filename = dir + names[i];
			cout << "processing image " << filename << endl;
			array2d<rgb_pixel> img;
			cv::Mat src = cv::imread(filename.c_str());
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

			std::vector<cv::Mat> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				cv::Rect rect;
				rect.x = dets[j].left();
				rect.y = dets[j].top();
				rect.width = dets[j].right() - dets[j].left();
				rect.height = dets[j].bottom() - dets[j].top();
				cv::Mat shape = sp(avg, rect);
				
				shapes.push_back(shape);
			}

			for (int j = 0; j < shapes.size(); j++) {
				cv::Mat res = shapes[j];
				for (int k = 0; k < 68; k++) {
					int tempx = res.at<float>(0, k);
					int tempy = res.at<float>(1, k);
					cv::circle(src, cv::Point(tempx, tempy), 2, cv::Scalar(255, 255, 255));
				}
			}
			for (int k = 0; k < dets.size(); k++) {
				cv::rectangle(src, cv::Point(dets[k].left(), dets[k].top()),
					cv::Point(dets[k].right(), dets[k].bottom()), cv::Scalar(255, 0, 0));
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


void face_landmark1()
{
	try
	{
		//frontal_face_detector detector = get_frontal_face_detector();
		frontal_face_detector detector;
		deserialize("frontface.dat") >> detector;
		dlib1::shape_predictor sp;
		//deserialize("D:/data/shape_predictor_68_face_landmarks.dat") >> sp;
		//save_ft("D:/data/1.yaml", sp);
		//return;
		sp = load_ft<dlib1::shape_predictor>("D:/data/1.yaml");
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
				for (int i = 0; i < 68; i++) {
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






int main() {
	//freopen("cv.txt", "w", stdout);
	face_landmark();
}

