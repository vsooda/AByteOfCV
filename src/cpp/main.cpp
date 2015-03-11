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

//¹âÕÕµÄ²ßÂÔÐèÒª¸ÄÉÆ
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


void tpsWarpIllum() {
    srand (time(NULL));
    int num = 50;
    int filenum = 600;
    //vector<int> det(68 * 2);
    vector<vector<int> > dets;
    vector<cv::Mat> images;
    FILE* fin = fopen("/home/sooda/data/face1/result/11.txt", "r");
    char filename[80];
    for(int i = 0; i < filenum; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        if(cnt <= 0) {
            break; 
        }

        std::cout << i << " "  << filename << std::endl;
        cv::Mat img = cv::imread(filename);
        int tempx, tempy;
        vector<int> det(68*2);
        for(int j = 0; j < 68; j++) {
            //fscanf(fin, "(%d, %d)%*c", &tempx, &tempy);
            fscanf(fin, "%d %d%*c", &tempx, &tempy);
            //cv::circle(img, cv::Point(tempx, tempy), 2, cv::Scalar(255, 0, 255));
            tempx = tempx * 2;
            tempy = tempy * 2;
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
            //¿¿
        }
        dets.push_back(det);
        //std::cout << i << " " << filename << std::endl;
        images.push_back(img);
        //drawCounter(img, det);
        cv::Mat pointSet;
    }

    //10 transform to 16
    for(int i = 0; i < filenum; i++) {
        
        int matchSize = 68;
        cv::Mat transformingImage, result;
        int transIndex = 501;
        //int transIndex = rand() % filenum + 1;
        //int transIndex = 10;
        //int targetIndex = i;
        int targetIndex = rand() % filenum + 1; 
        std::cout << "trans: " << transIndex << " target: " << targetIndex << std::endl;
        images[transIndex].copyTo(transformingImage);
        images[targetIndex].copyTo(result);
        cv::Mat shape1(matchSize, 2, CV_32F);
        cv::Mat shape2(matchSize, 2, CV_32F);
        vector<int> det1 = dets[transIndex];
        vector<int> det2 = dets[targetIndex];
        imshow("src", transformingImage);
        imshow("target", result);
        for(int i = 0; i < matchSize; i++) {
            shape1.at<float>(i, 0) = det1[2 * i];
            shape1.at<float>(i, 1) = det1[2*i + 1];
            shape2.at<float>(i, 0) = det2[2 * i];
            shape2.at<float>(i, 1) = det2[2*i + 1];
        }
        Ptr<cv::ThinPlateSplineShapeTransformer> tps = createThinPlateSplineShapeTransformer();
        tps->estimationTransformation1(shape2, shape1);
        tps->warpImage(transformingImage, result); 
        imshow("warpResult", result);
        //¿¿¿¿
        cv::Mat mask2 = getMask(result, det2);
        //imshow("mask2", mask2);
        cv::Mat extra = imread("sal.jpg", 0);
        //imshow("img", images[targetIndex]);
        customCV::IllumTransform::Params params(30, 0.2, extra, mask2);
       // cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create(1, params);
        //cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create();
        customCV::Makeup::Params makeupParams(params);
		cv::Ptr<customCV::Makeup> illumMethod = customCV::Makeup::create(customCV::Makeup::MAKEUP_ILLUM_TRANSFORM, makeupParams);
        cv::Mat dst = illumMethod->apply(images[targetIndex], result);
        //imshow("dst", dst);
        //for(int i = 0; i < mask2.cols; i++) {
        //    for(int j = 0; j < mask2.rows; j++) {
        //        if(mask2.at<uchar>(j, i) == 0){
        //            dst.at<cv::Vec3b>(j, i)[0] = 0;
        //            dst.at<cv::Vec3b>(j, i)[1] = 0;
        //            dst.at<cv::Vec3b>(j, i)[2] = 0;
        //        }
        //    }
        //}
        imshow("dst", dst);
        cv::waitKey();
    }
}

void tps_test() {
    int matchSize = 68;
    cv::Mat src, dst, result;
    //src = imread("ref2.png");   
    src = imread("ref4.png");   
    dst = imread("q4.jpg");
    result = src.clone();

    FILE *fin = fopen("ref4.txt", "r");

    cv::Mat shape1(matchSize, 2, CV_32F);
    cv::Mat shape2(matchSize, 2, CV_32F);
    for(int i = 0; i < matchSize; i++) {
        float x, y;
        fscanf(fin, "%f %f ", &x, &y); 
        cout << i << " " << x << " " << y << endl;
        cv::circle(src, cv::Point(x, y), 1, cv::Scalar(255, 0, 255));
        shape1.at<float>(i, 0) = x;
        shape1.at<float>(i, 1) = y;
    }
    fclose(fin);

    fin = fopen("q4.txt", "r");

    for(int i = 0; i < matchSize; i++) {
        float x, y;
        fscanf(fin, "%f %f ", &x, &y); 
        cv::circle(dst, cv::Point(x, y), 1, cv::Scalar(255, 0, 255));
        cout << i << " " << x << " " << y << endl;
        shape2.at<float>(i, 0) = x;
        shape2.at<float>(i, 1) = y;
    }

    Ptr<cv::ThinPlateSplineShapeTransformer> tps = createThinPlateSplineShapeTransformer();
    tps->estimationTransformation1(shape2, shape1);
    tps->warpImage(result, result);
    cv::Mat depth = cv::imread("ref4.pgm", 0);
    tps->warpImage(depth, depth);
    imwrite("ref4_warp.png", result);
    imwrite("ref4_warp.pgm", depth);
    imshow("src", src);
    imshow("dst", dst);

    imshow("depth", depth);
    imshow("warpResult", result);
    cv::waitKey();
    //¿¿¿¿
    //cv::Mat mask2 = getMask(result, det2);
}

void tpsWarpIllum1() {
    int num = 50;
    int filenum = 60;
    //vector<int> det(68 * 2);
    vector<vector<int> > dets;
    vector<cv::Mat> images;
    FILE* fin = fopen("/home/sooda/data/self/1.txt", "r");
    string basename = "/home/sooda/data/self/train/";
    int landmarknum = 68;
    char filename[80];
    for(int i = 0; i < filenum; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        if(cnt <= 0) {
            break; 
        }
        string fullname = basename + filename;
        cv::Mat img = cv::imread(fullname.c_str());
        int tempx, tempy, tempw, temph;
        fscanf(fin, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        vector<cv::Point> landmark;
        vector<int> det(68*2);
        for(int j = 0; j < landmarknum; j++) { 
            int x, y;
            fscanf(fin, "%d %d%*c", &x, &y);
            det[j * 2] = x;
            det[j*2 + 1] = y;
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255));
        }

        std::cout << i << " "  << fullname << std::endl;
        dets.push_back(det);
        images.push_back(img);
        cv::Mat pointSet;
    }

    //10 transform to 16
    for(int i = 5; i < filenum; i++) {
        
        int matchSize = 68;
        cv::Mat transformingImage, result;
        int transIndex = 45;
        //int transIndex = 10;
        int targetIndex = i;
        images[transIndex].copyTo(transformingImage);
        images[targetIndex].copyTo(result);
        cv::Mat shape1(matchSize, 2, CV_32F);
        cv::Mat shape2(matchSize, 2, CV_32F);
        vector<int> det1 = dets[transIndex];
        vector<int> det2 = dets[targetIndex];
        imshow("src", transformingImage);
        imshow("target", result);
        waitKey();
        for(int i = 0; i < matchSize; i++) {
            shape1.at<float>(i, 0) = det1[2 * i];
            shape1.at<float>(i, 1) = det1[2*i + 1];
            shape2.at<float>(i, 0) = det2[2 * i];
            shape2.at<float>(i, 1) = det2[2*i + 1];
        }
        Ptr<cv::ThinPlateSplineShapeTransformer> tps = createThinPlateSplineShapeTransformer();
        tps->estimationTransformation1(shape2, shape1);
        tps->warpImage(transformingImage, result); 
        imshow("warpResult", result);
        //¿¿¿¿
        cv::Mat mask2 = getMask(result, det2);
        imshow("mask2", mask2);
        //cv::Mat extra = imread("sal.jpg", 0);
        imshow("img", images[targetIndex]);
        //customCV::IllumTransform::Params params(30, 0.2, extra, mask2);
        //cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create(1, params);
        cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create();
        cv::Mat dst = illumMethod->apply(images[targetIndex], result);
        //imshow("dst", dst);
        for(int i = 0; i < mask2.cols; i++) {
            for(int j = 0; j < mask2.rows; j++) {
                if(mask2.at<uchar>(j, i) == 0){
                    dst.at<cv::Vec3b>(j, i)[0] = 0;
                    dst.at<cv::Vec3b>(j, i)[1] = 0;
                    dst.at<cv::Vec3b>(j, i)[2] = 0;
                }
            }
        }
        imshow("dst", dst);
        cv::waitKey();
    }
}

void showPnp() {
    int landmarknum = 7;
    string jpgname = "/home/sooda/study/HeadPosePnP/Angelina_Jolie/Angelina_Jolie_0001.jpg";
    FILE* fin = fopen("/home/sooda/study/HeadPosePnP/Angelina_Jolie/Angelina_Jolie_0001.txt", "r");
    cv::Mat img = cv::imread(jpgname.c_str());
    if(fin == NULL ) {
        std::cout << "open faild" << std::endl;
        return ; 
    }
    char filename[80];
    for(int i = 0; i < landmarknum; i++) {
        int x, y;
        fscanf(fin, "%d %d", &x, &y);
        char msg[20];
        sprintf(msg, "%d", i);
        putText(img, msg, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255)); 
    }
    cv::imshow("img", img);
    cv::waitKey();
}

void showHelenDatabase() {
    int filenum = 50;
    int landmarknum = 194;
    string basename = "/home/sooda/data/helen/";
    FILE* fin = fopen("/home/sooda/data/helen/annotation_dlib.txt", "r");
    char filename[80];
    cv::namedWindow("img", 0);
    for(int i = 0; i < filenum; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        string fullname = basename + filename;
        cv::Mat img = cv::imread(fullname.c_str());
        int tempx, tempy, tempw, temph;
        fscanf(fin, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        vector<cv::Point> landmark;
        for(int j = 0; j < landmarknum; j++) { 
            int x, y;
            fscanf(fin, "%d %d%*c", &x, &y);
            char msg[20];
            sprintf(msg, "%3d", j%100);
            putText(img, msg, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255)); 
            //cv::circle(img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255));
            //cv::imshow("img", img);
            //cv::waitKey();
        }
        std::cout << fullname << std::endl;
        cv::rectangle(img, cv::Point(tempx, tempy), cv::Point(tempw, temph), cv::Scalar(255, 255, 0));
        //cv::rectangle(img, cv::Point(tempy, tempx), cv::Point(temph, tempw), cv::Scalar(255, 0, 0));
        cv::imshow("img", img);
        cv::waitKey();
        imwrite("an.jpg", img);
    }

}

void showLfpwDatabase() {
    int filenum = 50;
    //int landmarknum = 68;
    int landmarknum = 7;
    string basename = "/home/sooda/data/lfpw/trainset/";
    FILE* fin = fopen("/home/sooda/data/lfpw/lfpw_train_7.txt", "r");
    char filename[80];
    for(int i = 0; i < filenum; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        string fullname = basename + filename;
        int tempx, tempy, tempw, temph;
        fscanf(fin, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        vector<cv::Point> landmark;
        cv::Mat img = cv::imread(fullname.c_str());
        for(int j = 0; j < landmarknum; j++) { 
            int x, y;
            fscanf(fin, "%d %d%*c", &x, &y);
            cv::circle(img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255));
            std::cout << x << " " << y << std::endl;
        }
        std::cout << fullname << std::endl;
        cv::rectangle(img, cv::Point(tempx, tempy), cv::Point(tempw, temph), cv::Scalar(255, 255, 0));
        //cv::rectangle(img, cv::Point(tempy, tempx), cv::Point(temph, tempw), cv::Scalar(255, 0, 0));
        cv::imshow("img", img);
        cv::waitKey();
    }
}

void showSelfDatabase() {
    int filenum = 200;
    //int landmarknum = 68;
    int landmarknum = 68;
    string basename = "/home/sooda/data/self/trainset/";
    FILE* fin = fopen("/home/sooda/data/self/self_dlib.txt", "r");
    char filename[80];
    for(int i = 0; i < filenum; i++) {
        int cnt = fscanf(fin, "%s%*c", filename);
        string fullname = basename + filename;
        int tempx, tempy, tempw, temph;
        fscanf(fin, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        vector<cv::Point> landmark;
        cv::Mat img = cv::imread(fullname.c_str());
        for(int j = 0; j < landmarknum; j++) { 
            int x, y;
            fscanf(fin, "%d %d%*c", &x, &y);
            char msg[20];
            sprintf(msg, "%3d", j%100);
            putText(img, msg, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 0, 255)); 
            //cv::circle(img, cv::Point(x, y), 2, cv::Scalar(255, 0, 255));
            std::cout << x << " " << y << std::endl;
        }
        std::cout << fullname << std::endl;
        cv::rectangle(img, cv::Point(tempx, tempy), cv::Point(tempw, temph), cv::Scalar(255, 255, 0));
        //cv::rectangle(img, cv::Point(tempy, tempx), cv::Point(temph, tempw), cv::Scalar(255, 0, 0));
        cv::imshow("img", img);
        cv::waitKey();
        cv::imwrite("template.jpg", img);
    }
}


int main() {
    showPnp();
    return 0;
    tps_test();
    return 0;
   // showSelfDatabase();
   // showLfpwDatabase();
   // showHelenDatabase();
   tpsWarpIllum();
   //tpsWarpIllum1();
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
