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
	T x; FileStorage f(fname, FileStorage::READ);
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
		//frontal_face_detector detector = get_frontal_face_detector();
		frontal_face_detector detector;
		deserialize("frontface.dat") >> detector;
		shape_predictor sp;
		deserialize("D:/data/shape_predictor_68_face_landmarks.dat") >> sp;
		//save_ft("D:/data/1.yaml", sp);
		//return;

		std::vector<string> names;
		string dir;
		int cnt = readDir("D:/data/*.jpg", names, dir);
		for (int i = 0; i < cnt; i++) {
			string filename = dir + names[i];
			cout << "processing image " << filename << endl;
			array2d<rgb_pixel> img;
			cv::Mat src = cv::imread(filename.c_str());
			dlib::cv_image<rgb_pixel> *pimg = new dlib::cv_image<rgb_pixel>(src);
			assign_image(img, *pimg);

			//load_image(img, filename.c_str());
			pyramid_up(img);

			std::vector<dlib::rectangle> dets = detector(img);

			std::vector<full_object_detection> shapes;
			for (unsigned long j = 0; j < dets.size(); ++j)
			{
				full_object_detection shape = sp(img, dets[j]);
				shapes.push_back(shape);
			}


			for (int j = 0; j < shapes.size(); j++) {
				full_object_detection res = shapes[j];
				for (int i = 0; i < 68; i++) {
					point pt = res.part(i);
					int tempx = pt.x() / 2;
					int tempy = pt.y() / 2;
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


void face_detect() {
	try
	{
		frontal_face_detector detector;
		//detector = get_frontal_face_detector();

		// Loop over all the images provided on the command line.
		std::vector<string> names;
		string dir;
		int cnt = readDir("D:/data/*.jpg", names, dir);
		for (int i = 0; i < cnt; i++) {
			string filename = dir + names[i];
			cout << "processing image " << filename << endl;
			array2d<unsigned char> img;
			//load_image(img, filename);
			// Make the image bigger by a factor of two.  This is useful since
			// the face detector looks for faces that are about 80 by 80 pixels
			// or larger.  Therefore, if you want to find faces that are smaller
			// than that then you need to upsample the image as we do here by
			// calling pyramid_up().  So this will allow it to detect faces that
			// are at least 40 by 40 pixels in size.  We could call pyramid_up()
			// again to find even smaller faces, but note that every time we
			// upsample the image we make the detector run slower since it must
			// process a larger image.
			pyramid_up(img);

			// Now tell the face detector to give us a list of bounding boxes
			// around all the faces it can find in the image.
			std::vector<dlib::rectangle> dets = detector(img);

			cout << "Number of faces detected: " << dets.size() << endl;
			// Now we show the image on the screen and the face detections as
			// red overlay boxes.

			cout << "Hit enter to process the next image..." << endl;
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


void trainLandmark()
{
	try
	{
		const std::string faces_directory = "D:/data/";

		dlib::array<array2d<unsigned char> > images_train, images_test;
		std::vector<std::vector<full_object_detection> > faces_train, faces_test;

		//		load_image_dataset(images_train, faces_train, faces_directory + "/training_with_face_landmarks.xml");
		//		load_image_dataset(images_test, faces_test, faces_directory + "/testing_with_face_landmarks.xml");

		shape_predictor_trainer trainer;

		trainer.set_oversampling_amount(300);

		trainer.set_nu(0.05);
		trainer.set_tree_depth(2);

		trainer.be_verbose();

		// Now finally generate the shape model
		shape_predictor sp = trainer.train(images_train, faces_train);

		cout << "mean training error: " <<
			test_shape_predictor(sp, images_train, faces_train, get_interocular_distances(faces_train)) << endl;

		cout << "mean testing error:  " <<
			test_shape_predictor(sp, images_test, faces_test, get_interocular_distances(faces_test)) << endl;

		// Finally, we save the model to disk so we can use it later.
		serialize("sp.dat") << sp;
	}
	catch (exception& e)
	{
		cout << "\nexception thrown!" << endl;
		cout << e.what() << endl;
	}
}





int main() {
	//trainLandmark();
	//return 0;
	//face_detect();
	//return 0;
	face_landmark();
}