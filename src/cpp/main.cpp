#include <iostream>
#include <string>
#include "photoAlgo.h"
#include "opencv2/photo/photo.hpp"
#include "test.h"
using namespace std;
using namespace cv;

void illumTransform_test() {
	IllumtrasformTest illum;
	illum.apply("../data/022.jpg", NULL, "../data/std_512.jpg");
}

void guideIllum_test() {
	IllumtrasformTest guide;
	guide.setAlgorithmType(customCV::IllumTransform::GUIDE_GUIDE_FILTER);
	cv::Mat dst = guide.apply("../data/022.jpg", "../data/b.bmp", "../data/std_512.jpg", "../data/stdsal.jpg");
}

//int readDir(string path, vector<string> &names, string& dir) {
//	//vector<string> names;
//	dir = path.substr(0, path.find_last_of("\\/") + 1);
//	names.clear();
//	names.reserve(10000);
//	WIN32_FIND_DATAA fileFindData;
//	HANDLE hFind = ::FindFirstFileA(path.c_str(), &fileFindData);
//	if (hFind == INVALID_HANDLE_VALUE) {
//		return 0;
//	}
//
//	do{
//		if (fileFindData.cFileName[0] == '.')
//			continue; // filter the '..' and '.' in the path
//		if (fileFindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
//			continue; // Ignore sub-folders
//		names.push_back(fileFindData.cFileName);
//	} while (::FindNextFileA(hFind, &fileFindData));
//	FindClose(hFind);
//	return (int)names.size();
//}
//
//void guideIllumTestBatch() {
//	vector<string> names;
//	string dir;
//	int cnt = readDir("D:/wkdir/skin/*.png", names, dir);
//	for (int i = 0; i < cnt; i++) {
//		string filename = dir + names[i];
//		IllumtrasformTest guide;
//		guide.setAlgorithmType(customCV::IllumTransform::GUIDE_GUIDE_FILTER);
//		cv::Mat dst = guide.apply(filename.c_str(), "../data/b.bmp", "../data/std_512.jpg", "../data/stdsal.jpg");
//		string savename = "result/";
//		savename = savename + names[i];
//		imwrite(savename.c_str(), dst);
//	}
//}
//
//
void quilt_test(char* filename, char* save_name, int size, int w, int niter) {
	QuiltTest quilt;
	quilt.setExtraParams(512, 40, 2);
	quilt.apply("../data/1.bmp");
}

void gcoTest() {
	QuiltTest gco;
	gco.setAlgorithmType(customCV::Quilting::QUILT_GRAPHCUT);
	gco.apply("../data/pp2.jpg", "../data/mask512.bmp", "../data/5_512.png");
}

void salancy_test()
{
	SalancyTest salancy;
	salancy.setAlgorithmType(customCV::Salancy::SALANCY_CMM_ILLUM);
	salancy.apply("../data/i3.png", "../data/mask512.bmp", "../data/std_512.jpg");
}

void inpaint_test() {
	InpaintTest inpaint;
	inpaint.apply("../data/i3.png", "../data/dd.png");
}

void epdfilter_test() {
	EpdfilterTest epd;
	epd.apply("../data/i3.png");
}

//光照的策略需要改善
void makeup_test() {
	MakeupTest makeup;
	makeup.apply("../data/5_512.jpg", "../data/pmask1024.jpg", "../data/mk.png", "../data/stdsal.jpg");
}

void skin_test() {
	SkinDetectorTest skindetector;
	skindetector.apply("../data/i3.jpg");
}

void tongmap_test() {
	TonemapTest tonemap;
	tonemap.apply("../data/2.jpg");
}

void colorTransform_test() {
	cv::Mat src, q, dst, mask;
	src = imread("../data/038.jpg");
	q = imread("../data/std_512.jpg");
	mask = imread("../data/b.bmp", 0);
	dst = customCV::colorTransform(src, q);
	imshow("dst", dst);
	waitKey();
}

int main() {
	//guideIllumTestBatch();
	//return 0;
//	illumTransform_test();
	guideIllum_test();
//	quilt_test("../data/1.bmp", "../data/dcc.png", 512, 40, 2);
//	gcoTest();
//	salancy_test();
//	inpaint_test();
//	epdfilter_test();
//	makeup_test();
//	skin_test();
//	tongmap_test();
	
}
