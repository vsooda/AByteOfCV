#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <time.h>
#include "inpaint_imp.h"
#include "inpaint.h"
#include "quilt.h"
#include <stdarg.h>
#include "util.h"
#include <time.h>
#include "illumination.h"


using namespace std;
using namespace cv;

void LBP (IplImage *src,IplImage *dst) ;


int addImage(const Mat& img1, const Mat& img2, const Mat &mask, float alphaX, Mat& dst) {
	if(img1.cols != img2.cols || img1.rows != img2.rows || img1.cols != mask.cols || img1.rows != mask.rows 
		|| img1.cols != dst.cols || img1.rows != dst.rows) {
		cout << "invalid input in addImage: the size of image is diffirent" << endl;
		return -1;
	}
	int col = img1.cols;
	int row = img1.rows;
	for(int i = 0; i < col; i++) {
		for(int j = 0; j < row; j++) { 
			if(mask.at<uchar>(j,i) > 10) {
				float alpha = 1.0 * img1.at<cv::Vec4b>(j,i)[3] / 255.0 * alphaX;
				dst.at<cv::Vec3b>(j,i)[0] = img1.at<cv::Vec4b>(j,i)[0] * alpha + img2.at<cv::Vec3b>(j,i)[0] * (1-alpha);
				dst.at<cv::Vec3b>(j,i)[1] = img1.at<cv::Vec4b>(j,i)[1] * alpha + img2.at<cv::Vec3b>(j,i)[1] * (1-alpha);
				dst.at<cv::Vec3b>(j,i)[2] = img1.at<cv::Vec4b>(j,i)[2] * alpha + img2.at<cv::Vec3b>(j,i)[2] * (1-alpha);
			}
		}
	}
}

int addImageWithoutAlpha(const Mat& img1, const Mat& img2, const Mat &mask, Mat& dst) {
	if(img1.cols != img2.cols || img1.rows != img2.rows || img1.cols != mask.cols || img1.rows != mask.rows 
		|| img1.cols != dst.cols || img1.rows != dst.rows) {
		cout << "invalid input in addImage: the size of image is diffirent" << endl;
		return -1;
	}
	int col = img1.cols;
	int row = img1.rows;
	for(int i = 0; i < col; i++) {
		for(int j = 0; j < row; j++) { 
			if(mask.at<uchar>(j,i) > 10) {
				float alpha = 1.0 * img1.at<cv::Vec4b>(j,i)[3] / 255.0 * 0.5; 
				dst.at<cv::Vec3b>(j,i)[0] = img1.at<cv::Vec4b>(j,i)[0] * alpha + img2.at<cv::Vec3b>(j,i)[0] * (1-alpha);
				dst.at<cv::Vec3b>(j,i)[1] = img1.at<cv::Vec4b>(j,i)[1] * alpha + img2.at<cv::Vec3b>(j,i)[1] * (1-alpha);
				dst.at<cv::Vec3b>(j,i)[2] = img1.at<cv::Vec4b>(j,i)[2] * alpha + img2.at<cv::Vec3b>(j,i)[2] * (1-alpha);
			}
		}
	}
} 


//获取轮廓，并将其画出来--这里的src应该是二值图像
void detectAndDrawContours(Mat& src) {
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
	int idx = 0;
	for( ; idx >= 0; idx = hierarchy[idx][0] )
	{
		Scalar color(50, 50, 50);
		drawContours( src, contours, idx, color, 3, 8, hierarchy );
	}
	
}

//转化为灰度图像，再二值化
Mat binarize(Mat img, int thresh_value) {
	Mat gray;
	Mat thresholded; 
	cvtColor(img, gray, CV_BGRA2GRAY);
//	threshold(gray,thresholded, thresh_value, 255, THRESH_BINARY_INV);
	threshold(gray,thresholded, thresh_value, 255, THRESH_BINARY);
//	imshow("thresholded_orig", thresholded);
	return thresholded;
}

void dealMouthAndEye(Mat& src_img) {
	Mat add_img = imread("add1.png", -1);
	//cvtColor(add_img, add_img, CV_BGRA2BGR); 
	//imshow("add1", add_img);
	Mat thresholded;	
	thresholded = binarize(add_img, 190);
	cout << thresholded.channels() << endl;
	//imshow("mouth", thresholded);
	addImage(add_img, src_img, thresholded, 0.8, src_img);	
	//addImageWithoutAlpha(add_img, src_img, thresholded, src_img);
}

void dealFace(Mat& src_img) {
	Mat add_img = imread("add4.png", -1);
	Mat thresholded;	
	thresholded = binarize(add_img, 190);
	addImage(add_img, src_img, thresholded, 0.3, src_img);
	//addImageWithoutAlpha(add_img, src_img, thresholded, src_img);
}

void showResult() {
	Mat src_img = imread("src.png");
	imshow("原始图片", src_img);
	dealFace(src_img);
	imshow("脸部处理后", src_img);
	dealMouthAndEye(src_img);

//	addWeighted( src, 0.5, add1, 0.5, 0.0, src);

	imshow("最终结果", src_img);
}

void testBeauty() {
	for(int i = 2; i <= 8; i++) {
		char filename[30];
		sprintf(filename, "00000000000000000%d_head.png", i);
		Mat src_img = imread(filename);
	//	dealFace(src_img);
		dealMouthAndEye(src_img);
		char outputName[50];
		memset(outputName, 0, sizeof(outputName));
		sprintf(outputName, "result/00000000000000000%d_head.png", i);
		imwrite(outputName, src_img);
	}
} 

void filter_test() {
    Mat image, result;
	image = imread("building.jpg");
    cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
    // assigns kernel values
    kernel.at<float>(1,1)= 5.0;
    kernel.at<float>(0,1)= -1.0;
    kernel.at<float>(2,1)= -1.0;
    kernel.at<float>(1,0)= -1.0;
    kernel.at<float>(1,2)= -1.0;
    //filter the image
    cv::filter2D(image,result,image.depth(),kernel);
	cout << image.size() << endl;
	cout << result.size() << endl;
	imshow("image", image);
	imshow("result", result);
	waitKey();
}


void paste(Mat &bg, Mat obj, Rect rect) {
	Mat imageRoi = bg(rect);
	addWeighted(bg(rect), 0, obj, 1, 0., imageRoi);
}


Mat synthfun(Mat texture, int width, int niter) {
    Size sz = texture.size();
    cout << sz.width << " " << sz.height << endl;
    return texture;
}

void inpaint_test() {
	clock_t a, b;
	
	Mat img, mask, result, result2;
	img = imread("1.png");
	mask = imread("mask1.jpg");
	mask = binarize(mask, 100);
	a = clock();
	cv::inpaint(img, mask, result, 5, CV_INPAINT_TELEA);
	//customCV::inpaint(img, mask, result, 5, CV_INPAINT_TELEA);
	//customCV1::inpaint(img, mask, result, 3);
	b = clock();
	cout << (b - a)*1000.0 / CLOCKS_PER_SEC << endl;
	imshow("result", result);
	cv::waitKey();
}

void quilt_test(char* filename, char* save_name, int size, int w, int niter) {
	//cv::Mat dst(512,512,CV_32FC3,cv::Scalar(0, 0, 0));
	cv::Mat dst(size,size,CV_32FC3,cv::Scalar(0, 0, 0));
	cv::Mat src = imread(filename);
	cv::Mat texture;	
	src.convertTo(texture, CV_32FC3);
	cvConvertScale(&((IplImage)texture), &((IplImage)texture), 1.0/255.0);
	customCV::quilt(texture, dst, w, niter);
	cout << texture.at<Vec3f>(10, 10) << endl;
	cvConvertScale(&((IplImage)dst), &((IplImage)dst), 255.0);
	dst.convertTo(dst, CV_8UC3);
	imwrite(save_name, dst);
	imshow("src", src);
	imshow("texture", texture);
	imshow("dst", dst);
	//imwrite("dst_5.jpg", dst);
	waitKey();
}


void quilt_combine_test2(char* bg_name, char* src_name, char* mask_name, char* save_name) {
	cv::Mat bg = imread(bg_name);
	cv::Mat src = imread(src_name);
	cv::Mat mask = imread(mask_name);
	
	//mask = binarize(mask, 100); 

	int width = bg.cols;
	int height = bg.rows;

	cv::resize(mask, mask, cv::Size(width, height));
	cv::resize(src, src, cv::Size(width, height));
	mask.convertTo(mask, CV_32FC3);
	cvConvertScale(&((IplImage)mask), &((IplImage)mask), 1.0 / 255.0);
	bg.convertTo(bg, CV_32FC3);
	src.convertTo(src, CV_32FC3); 

	cv::Mat mask3;
	cv::Mat maskMats[] = {mask, mask, mask};
	cv::merge(maskMats, 3, mask3);	

	//cv::Mat oneMats = cv::Mat(width, height, CV_8UC3, Scalar(1, 1,1));
	cv::Mat oneMat3;
	cv::Mat oneMat = cv::Mat::ones(width, height, CV_32F);
	cv::Mat oneMats[] = {oneMat, oneMat, oneMat};
	cv::merge(oneMats, 3, oneMat3);
	//bg.convertTo(bg, CV_8UC3);
	//src.convertTo(src, CV_8UC3);
	//mask3.convertTo(mask3, CV_8UC3);
	cout << oneMat3.size() << endl;
	cout << mask.size() << endl;
	cout << src.size() << endl;
	cout << bg.size() << endl;
	cout << mask.type() << endl;
	cout << src.type() << endl;

	cv::Mat result, temp1, temp2;
	//temp1 = bg.mul(oneMat3 - mask);
	//temp2 = src.mul(mask);
	result = bg.mul(oneMat3 - mask) + src.mul(mask);
	result.convertTo(result, CV_8UC3);
	imshow("mask", mask);
	imshow("result", result);
	imwrite(save_name, result);
	cv::waitKey();	
}

void norm_test() {
	cv::Mat src, dst, dst2;
	src = imread("building.jpg");
	cv::normalize(src, dst, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC3);
	cv::normalize(dst, dst2, 0, 255, cv::NORM_MINMAX,  CV_8UC3);
	imshow("src", src);
	imshow("dst", dst);
	imshow("dst2", dst2);
	cv::waitKey();
}

//对mat操作样例
void Mat_test() {
	Mat A = Mat::eye(10, 10, CV_32S);
	Mat B = A(Range::all(), Range(1,6));  //opencv中mat下标是从0开始的。range（1，3）只包括1，2
	Mat C = Mat::ones(10, 10, CV_32S);
	C.row(5).copyTo(A.row(5));
	Mat D = B.diag(0);
	std::cout << A << std::endl;
	std::cout << B.t() << std::endl;
	std::cout << D << std::endl;
}


//处理三通道，返回其方差
Mat calcMatSdv(Mat src, int size) {
	Mat src_padd;
	int padd_size = size / 2;
	int size_half = size / 2;

	copyMakeBorder(src, src_padd, padd_size, padd_size, padd_size, padd_size, BORDER_REPLICATE);
	//imshow("src_padd", src_padd);
	//waitKey();
	//cout << src_padd.size() << endl;

	int rows = src.rows, cols = src.cols;
	Mat sdv(rows, cols, CV_32FC3, Scalar(0, 0, 0));
	for(int i = size_half; i < rows+size_half; i++) {
		for(int j = size_half; j < cols+size_half; j++) {
			Mat roi;
			Rect rect(i-size_half, j-size_half, size, size);
			roi = src_padd(rect);
			Scalar m, d;
			//float m, d;
			//cout << m << endl;
			//d = m;
			meanStdDev(roi, m, d); 
			//cout << i << " " << j << endl;
			//cout << d.val[0] << endl;
			sdv.at<Vec3f>(i-size_half, j-size_half)[0] = d.val[0];
			sdv.at<Vec3f>(i-size_half, j-size_half)[1] = d.val[1];
			sdv.at<Vec3f>(i-size_half, j-size_half)[2] = d.val[2];
		}
	}
	return sdv;
}

void calcMatSdv_test() {
	Mat src_img = imread("test/i5.png");
	Mat sdv = calcMatSdv(src_img, 31);	
} 


void weightProcess(Mat & w) {
	w.convertTo(w, CV_8U);
	//dst_ = customCV::Erosion(dst, 1, 1);
	w = customCV::Dilation(w, 1, 0);
	threshold(w,w, 50, 255, THRESH_BINARY);
}




void getMaskPic(char* src_name, char* mask_name, char* save_name, int size) {
	Mat src = imread(src_name);
	Mat mask = imread(mask_name);
	resize(src, src, Size(size, size));
	resize(mask, mask, Size(size, size));
	threshold(mask, mask, 10, 255, THRESH_BINARY);
	Mat dst;
	src.copyTo(dst, mask);
	imshow("dst", dst);
	imwrite(save_name, dst);
	waitKey();
}

void auto_test(char* patchname, char* srcname, char* maskname) {
	quilt_test(patchname, "quilt.jpg", 512, 40, 1);
	IplImage * mask = cvLoadImage(maskname, CV_LOAD_IMAGE_GRAYSCALE);
	getMaskPic(srcname, maskname, "roi.jpg", 512);
	customCV::transform_test("roi.jpg", "quilt.jpg", "bg.jpg", maskname);
	quilt_combine_test2("bg.jpg", srcname, maskname, "ret.jpg");
}



void highpass_test() {
	
	Mat src, dst;
	src = imread("test/i3.png");
	cvtColor(src, src, CV_BGR2GRAY);
	//Laplacian(src, dst, CV_16S);
	//convertScaleAbs( dst, dst );
    Mat kernel(3,3,CV_32F,Scalar(-1));   
    // 分配像素置  
    kernel.at<float>(1,1) = 8;  
    filter2D(src,dst,src.depth(),kernel); 
	imshow("dst", dst);
	waitKey();
}



void LBP_test() {
	int histSize = 256;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat src, dst, b_hist;
	src = imread("test/i3.png");
	cvtColor(src, src, CV_BGR2GRAY);
	src.copyTo(dst);
	customCV::LBP(&(IplImage(src)), &(IplImage(dst)));
	calcHist( &dst, 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
	Mat dst_;
	//dst_ = customCV::Erosion(dst, 1, 1);
	dst_ = customCV::Dilation(dst, 1, 0);
	dst_ = customCV::Erosion(dst_, 1, 1);
	//cout << dst << endl;
	imshow("dst_", dst_);
	imshow("src", src);
	imshow("dst", dst);
	imshow("hist", b_hist);
	waitKey();
}

void illumination_test() {
	customCV::funcParam param;
	param.val[0] = 0.6;
	param.val[1] = 0.1;
	param.val[2] = 0.1;
	param.val[3] = 0.2;
	customCV::illumProcess("test/i3.png", "test/sstd.jpg", "test/rrt.png", "test/mask512.bmp", param);
}

int main() {
	//auto_test("test/a.jpg", "test/000000000000000005_head.png", "test/mmask.bmp");
	illumination_test();
}


