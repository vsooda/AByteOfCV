#include <iostream>
#include <string>
#include "photoAlgo.h"
#include "opencv2/photo/photo.hpp"
#include "test.h"
#include <stdio.h>
#include <vector>
#include "shape/shape_transformer.hpp"
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
	//salancy.setAlgorithmType(customCV::Salancy::SALANCY_CMM_ILLUM);
	//salancy.apply("../data/i3.png", "../data/mask512.bmp", "../data/std_512.jpg");
	salancy.setAlgorithmType(customCV::Salancy::SALANCY_CMM);
    salancy.apply("/home/sooda/data/face/result/image_0078.png");
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

void drawCounter(cv::Mat src, vector<int> det) {
    //1-17 ==> 0-16
    for(int i = 0; i < 16; i++) {
        int j = i + 1;
        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
        cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
    }
    //18-27 ==> 17-26
    for(int i = 17; i < 26; i++) {
        int j = i + 1;
        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
        cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
    }

    //1-18 ==> 0-17
    cv::Point pt1 = cv::Point(det[0], det[1]);
    cv::Point pt2 = cv::Point(det[2*17], det[2*17+1]);
    cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
    
    //27-17 ==> 26-16
    pt1 = cv::Point(det[2*26], det[2*26+1]);
    pt2 = cv::Point(det[2*16], det[2*16+1]);
    cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));

    
    cv::imshow("lined", src);
    cv::waitKey();
}

cv::Mat getCoutourPoint(vector<int> det) {
//    //1-17 ==> 0-16
//    for(int i = 0; i < 16; i++) {
//        int j = i + 1;
//        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
//        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
//        cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
//    }
//
//    //17-27 ==> 16-26
//    cv::Point pt1 = cv::Point(det[2*16], det[2*16+1]);
//    cv::Point pt2 = cv::Point(det[2*26], det[2*26+1]);
//    cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
//
//    //27-18 ==> 26-17
//    for(int i = 26; i > 17; i--) {
//        int j = i - 1;
//        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
//        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
//        cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
//    }
//
//    //18-1 ==>17-0 
//     pt1 = cv::Point(det[2*17], det[2*17+1]);
//     pt2 = cv::Point(det[0], det[1]);
//    cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
//    
//    cv::line(src, pt1, pt2, cv::Scalar(255, 0, 255));
//    return src;
    cv::Mat pointSet(27, 1, CV_32FC2);
    for(int i = 0; i < 17; i++) {
        pointSet.at<cv::Vec2f>(i, 0)[0] = det[2*i];
        pointSet.at<cv::Vec2f>(i, 0)[1] = det[2*i+1];
    }
    for(int i = 26; i >=17; i--) {
        pointSet.at<cv::Vec2f>(i, 0)[0] = det[2*i];
        pointSet.at<cv::Vec2f>(i, 0)[1] = det[2*i+1];
    }
    return pointSet;

}

cv::Mat getMask(cv::Mat img, vector<int> det) {
    cv::Mat mask(img.size(), CV_8U, Scalar(0));
    //1-17 ==> 0-16
    int lineValue = 128;
    for(int i = 0; i < 16; i++) {
        int j = i + 1;
        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
        cv::line(mask, pt1, pt2, cv::Scalar(128));
    }

    //17-27 ==> 16-26
    cv::Point pt1 = cv::Point(det[2*16], det[2*16+1]);
    cv::Point pt2 = cv::Point(det[2*26], det[2*26+1]);
    cv::line(mask, pt1, pt2, cv::Scalar(128));

    //27-18 ==> 26-17
    for(int i = 26; i > 17; i--) {
        int j = i - 1;
        cv::Point pt1 = cv::Point(det[2*i], det[2*i+1]);
        cv::Point pt2 = cv::Point(det[2*j], det[2*j+1]);
        cv::line(mask, pt1, pt2, cv::Scalar(128));
    }

    //18-1 ==>17-0 
     pt1 = cv::Point(det[2*17], det[2*17+1]);
     pt2 = cv::Point(det[0], det[1]);
    cv::line(mask, pt1, pt2, cv::Scalar(128));

    
    //fill
    for(int i = 0; i < mask.cols; i++) {
        for(int j = 0; j < mask.rows; j++) {
            if(mask.at<uchar>(j, i) == 128) {
                break;
            }
            else {
                mask.at<uchar>(j, i) = 255;
            }
        }
    }

    for(int j = 0; j < mask.rows; j++) {
        for(int i = 0; i < mask.cols; i++) {
            if(mask.at<uchar>(j, i) == 128) {
                break;
            }
            else {
                mask.at<uchar>(j, i) = 255;
            }
        }
    }
    
    for(int j = mask.rows-1; j >= 0; j--) {
        for(int i = mask.cols-1; i >= 0; i--) {
            if(mask.at<uchar>(j, i) == 128) {
                break;
            }
            else {
                mask.at<uchar>(j, i) = 255;
            }
        }
    }


    for(int i = mask.cols-1; i >= 0; i--) {
        for(int j = mask.rows-1; j >= 0; j--) {
            if(mask.at<uchar>(j, i) == 128) {
                break;
            }
            else {
                mask.at<uchar>(j, i) = 255;
            }
        }
    }

    mask = 255 - mask;
    
    return mask;
}


void showDetectResult() {
    int num = 50;
    //vector<int> det(68 * 2);
    vector<vector<int> > dets;
    vector<cv::Mat> images;
    FILE* fin = fopen("/home/sooda/data/face/result/1.txt", "r");
    char filename[80];
    for(int i = 0; i < 50; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        if(cnt <= 0) {
            break; 
        }

        std::cout << filename << std::endl;
        int tempx, tempy;
        vector<int> det(68*2);
        for(int j = 0; j < 68; j++) {
            fscanf(fin, "(%d, %d)%*c", &tempx, &tempy);
            if(tempx < 0) 
                tempx = 1;
            if(tempy < 0) 
                tempy = 1;
            if(tempx > 498) { 
                tempx = 498;
            }
            if(tempy > 598) {
                tempy = 598;
            }
            det[j * 2] = tempx / 2;
            det[j*2 + 1] = tempy / 2;
//            std::cout << tempx << " " << tempy << std::endl;
            //靠
        }
        dets.push_back(det);
        //std::cout << i << " " << filename << std::endl;
        cv::Mat img = cv::imread(filename);
        images.push_back(img);
        //drawCounter(img, det);
        cv::Mat pointSet;
        //pointSet = getCoutourPoint(det);
        //cv::Rect rect = boundingRect(pointSet);
        //cv::rectangle(img, rect, cv::Scalar(0, 0, 255));
        
        //cv::Mat mask = cv::Mat::zeros(img.size(), CV_8U);
        //pointSet.convertTo(pointSet, CV_32SC2);
        //drawContours(mask, pointSet, 1, Scalar(255), CV_FILLED);
        //imshow("mask", mask);
        //
    

        cv::Mat mask = getMask(img, det);

//        imshow("mask", mask);
//        cv::Mat sal;
//		img.convertTo(img, CV_32FC3, 1.0 / 255.0);
//		cv::Ptr<customCV::Salancy> salancyMethod = customCV::Salancy::create(0);
//		sal= salancyMethod->apply(img, mask, mask);
//		sal.convertTo(sal, CV_8UC1, 255);
//        imshow("sal", sal);
//

        //cv::Rect rect = boundingRect(det);
        for(int i = 0; i < 68; i++) {
            cv::circle(img, cv::Point(det[i*2], det[i*2+1]), 3, cv::Scalar(255, 0, 0)); 
        }
//        imshow("img", img);
 //       waitKey();
        //cv::imshow("result", img);
        //cv::waitKey(10);
    }

 //       vector<int> det;
 //       det = dets[i];
 //       for(int j = 0;j < 68; j++) {
 //           std::cout << det[2*j] << " " << det[j+1] << " ";
 //       }
 //       std::cout << std::endl;
 //   }
    //10 transform to 16
    for(int i = 0; i < 39; i++) {
        
        int matchSize = 68;
        cv::Mat transformingImage, result;
        int transIndex = 40;
        //int transIndex = 10;
        int targetIndex = i;
        images[transIndex].copyTo(transformingImage);
        images[targetIndex].copyTo(result);
        cv::Mat shape1(matchSize, 2, CV_32F);
        cv::Mat shape2(matchSize, 2, CV_32F);
        vector<int> det1 = dets[transIndex];
        vector<int> det2 = dets[targetIndex];
        for(int i = 0; i < 68; i++) {
            cv::circle(transformingImage, cv::Point(det1[i*2], det1[i*2+1]), 3, cv::Scalar(255, 0, 0)); 
            cv::circle(result, cv::Point(det2[i*2], det2[i*2+1]), 3, cv::Scalar(255, 0, 0)); 
        }
        imshow("src", transformingImage);
        imshow("target", result);
        for(int i = 0; i < matchSize; i++) {
            shape1.at<float>(i, 0) = det1[2 * i];
            shape1.at<float>(i, 1) = det1[2*i + 1];
            shape2.at<float>(i, 0) = det2[2 * i];
            shape2.at<float>(i, 1) = det2[2*i + 1];
        }
        //std::cout << det1.size() << " " << det2.size() << std::endl;
     //   FILE* fout = fopen("map.txt", "w");
     //   for(int i = 0; i < matchSize * 2; i++) {
     //       fprintf(fout, "%d ", det1[i]); 
     //   }
     //   fprintf(fout, "\n");
     //   for(int i = 0; i < matchSize * 2; i++) {
     //       fprintf(fout, "%d ", det2[i]); 
     //   }
     //   for(int i = 0; i < matchSize; i++) {
     //       std::cout << shape1.at<float>(i, 0)  << std::endl;
     //   }
        Ptr<cv::ThinPlateSplineShapeTransformer> tps = createThinPlateSplineShapeTransformer();
        //tps->estimationTransformation1(shape1, shape2);
        tps->estimationTransformation1(shape2, shape1);
        tps->warpImage(transformingImage, result); 
        imshow("result", result);
        cv::waitKey();
    }

}


int main() {
    showDetectResult();
	//guideIllumTestBatch();
	//return 0;
//	illumTransform_test();
//	guideIllum_test();
//	quilt_test("../data/1.bmp", "../data/dcc.png", 512, 40, 2);
//	gcoTest();
//	salancy_test();
//	inpaint_test();
//	epdfilter_test();
//	makeup_test();
//	skin_test();
//	tong:map_test();
}
