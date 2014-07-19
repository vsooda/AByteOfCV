#include "illumination.h"

namespace customCV {

	cv::Mat Symmetry(cv::Mat lhs, cv::Mat rhs) {
		cv::Mat ret;
		cv::Mat t1, t2;
		lhs.convertTo(t1, CV_32F);
		rhs.convertTo(t2, CV_32F);
		//ret = lhs - rhs;
		ret = t1 - t2;
		convertScaleAbs(ret, ret);
		ret.convertTo(ret, lhs.type());
		return ret;
	}


	void transform_test(char* src_filename, char* dst_filename, char* save_name = 0, char* mask_name = NULL) {
		IplImage* source = cvLoadImage(src_filename,CV_LOAD_IMAGE_COLOR);    
		IplImage* dst = cvLoadImage(dst_filename,CV_LOAD_IMAGE_COLOR);  
		IplImage* dstlab = cvCreateImage(cvGetSize(dst),dst->depth,dst->nChannels);    
		IplImage* res  = cvCreateImage(cvGetSize(dst),dst->depth,dst->nChannels);  
		IplImage * mask = cvLoadImage("test/mask512.bmp", CV_LOAD_IMAGE_GRAYSCALE);
    
		dstlab = cvCloneImage(dst);  
  
		//calculate average and standard derivation  
		CvScalar avg_src,avg_dst,std_src, std_dst;   
		cvAvgSdv(source,&avg_src,&std_src, mask);  
		cvAvgSdv(dstlab, &avg_dst, &std_dst, mask);  
		for(int i = 0; i < 4; i++) {
			std_src.val[i] = 1.0;
			std_dst.val[i] = 1.0;
		} 
		//transform  
		dstlab = customCV::Transform(dstlab,avg_src,avg_dst,std_src, std_dst, mask);  
		res = cvCloneImage(dstlab);  
		std::cout << cv::Scalar(avg_src) << std::endl;
		std::cout << cv::Scalar(avg_dst) << std::endl;

      
		customCV::cvShowManyImages("Color Transform",3, source, dst, res);  
		if(save_name != 0) {
			cvSaveImage(save_name, res);
		}
		cvWaitKey();  
	}
	
	cv::Mat doTransform(char* src_filename, char* dst_filename, char* save_name, char* mask_name) {
		cv::Mat src = cv::imread(src_filename);
		cv::Mat dst = cv::imread(dst_filename);
		cv::Mat mask = cv::imread(mask_name);
		cv::cvtColor(mask, mask, CV_BGR2GRAY);
		cv::Scalar avg_src,avg_dst,std_src, std_dst;
		meanStdDev(src, avg_src, std_src, mask);
		meanStdDev(dst, avg_dst, std_dst, mask);

		for(int i = 0; i < 4; i++) {
			std_src.val[i] = 1.0;
			std_dst.val[i] = 1.0;
		} 

		cv::Mat dstlab;
		dst.copyTo(dstlab);
		//transformMat(cv::Mat& src, cv::Scalar avg_src, cv::Scalar avg_dst, cv::Scalar std_src, cv::Scalar std_dst, cv::Mat mask)
		transformMat(dstlab, avg_src, avg_dst, std_src, std_dst, mask);
		customCV::cvShowManyImages("Color Transform",3, &(IplImage)src, &(IplImage)dst, &(IplImage)dstlab); 
		return dstlab;
	} 


 
void illumProcess(char * src1, char* src2, char* save_name, char* mask_name, customCV::funcParam fp) {

	float symParam = fp.val[0];
	float absParam = fp.val[1];
	float laplaceParam = fp.val[2];
	float logParam = fp.val[3];

	cv::Mat src_1 = cv::imread(src1);
	cv::Mat src_2;
	src_2 = doTransform(src1, src2, NULL, mask_name);


	cv::Mat dst;
	cv::Mat mask = cv::imread(mask_name);
	cvtColor(mask, mask, CV_BGR2GRAY);
	mask.convertTo(mask, CV_32F);
	cvConvertScale(&((IplImage)mask), &((IplImage)mask), 1.0 / 255.0);

	std::vector<cv::Mat> MatVec1, MatVec2;
	std::vector<cv::Mat> dstVec;
	//拆分为3通道
	cv::split(src_1,  MatVec1);
	cv::split(src_2, MatVec2);
	
	//add lbp feature
	cv::Mat src_gray, lbp_w;
	cvtColor(src_1, src_gray, CV_BGR2GRAY);
	src_gray.copyTo(lbp_w);
	customCV::LBP(&(IplImage(src_gray)), &((IplImage)lbp_w));
		
	mask.convertTo(mask, CV_32F);
	lbp_w.convertTo(lbp_w, CV_32F);
	//cvConvertScale(&((IplImage)lbp_w), &((IplImage)lbp_w), 1.0 / 255.0);
	normalize(lbp_w, lbp_w, 0., 1.0, cv::NORM_MINMAX);

	cv::Mat oneMat = cv::Mat::ones(src_1.cols, src_1.rows, CV_32F);

	cv::Mat laplacian_w;

	Laplacian(src_gray, laplacian_w, CV_16S);
	convertScaleAbs( laplacian_w, laplacian_w );

	//处理大块光照
	cv::Mat thresh;
	laplacian_w.copyTo(thresh);
	thresh = 255.0 - thresh;
	threshold(thresh,thresh, 252, 255, cv::THRESH_BINARY);
	//Dilation(thresh, 1, 1);
	thresh = Erosion(thresh, 3, 1);
//	thresh = Dilation(thresh, 3, 1);
	imshow("threash_lap", thresh);
	thresh.convertTo(thresh, CV_32F);
	thresh = thresh.mul(mask);
	threshold(thresh,thresh, 200, 255, cv::THRESH_BINARY);
	cvConvertScale(&((IplImage)thresh), &((IplImage)thresh), 1.0 / 255.0);
	//thresh.convertTo(thresh, CV_8U);
	//imshow("dd", thresh);




	laplacian_w.convertTo(laplacian_w, CV_32F);
	
	cv::Mat temp;
	laplacian_w.convertTo(temp, CV_8U);
	imshow("laplacian", temp);
	lbp_w.convertTo(temp, CV_8U);
	imshow("lbp", lbp_w);

	normalize(laplacian_w, laplacian_w, 0, 1.0, cv::NORM_MINMAX);

	cv::Mat logdiff;
	cv::Mat zeroMat = cv::Mat::zeros(src_1.cols, src_1.rows, CV_32F);
	

	for(int c = 0; c < 3; c++) {
		cv::Mat src_1c = MatVec1[c];
		cv::Mat src_2c = MatVec2[c];

		src_1c.convertTo(src_1c, CV_32F);
		src_2c.convertTo(src_2c, CV_32F);


		cv::Mat med1, med2; 
		src_1c.copyTo(med1);
		src_2c.copyTo(med2);

		cvSmooth(&((IplImage)src_1c), &((IplImage)med1), CV_BLUR, 31);
		cvSmooth(&((IplImage)src_2c), &((IplImage)med2), CV_BLUR, 31);

		cv::Mat sym, med_gray, med_flip;
		cv::Mat sym_w;
		med1.copyTo(med_gray);
		flip(med_gray, med_flip, 1);

		sym = Symmetry(med_gray, med_flip);
		sym.convertTo(sym, CV_32F);
		normalize(sym, sym_w, 0, 1.0, cv::NORM_MINMAX);

		sym.convertTo(sym, CV_8U);
		imshow("sym", sym);
		cv::waitKey();

		med1.convertTo(med1, CV_32F);
		med2.convertTo(med2, CV_32F);
		cv::Mat diff = med1 - med2;
		log(abs(diff), logdiff);
		logdiff = cv::max(zeroMat, logdiff);
		cv::Mat logdiff_w;
		normalize(logdiff, logdiff_w, 0, 1.0, cv::NORM_MINMAX);

		//cv::Mat w = sym_w * 0.6 + 0.1 + (1.0-laplacian_w)*0.1 + (1-logdiff_w)*0.2;
		cv::Mat w = sym_w * symParam + absParam + (1.0-laplacian_w)*laplaceParam + (1-logdiff_w)*logParam;
		cv::Mat diff_weight = diff.mul(w);
		cv::Mat dst_c = src_1c - diff_weight;

		//去除光块
		cv::Mat dst_flip;
		cv::Mat dst_;
		flip(dst_c, dst_flip, 1);

		//dst_c = (oneMat-thresh).mul(dst_c) + thresh.mul(0.7* dst_c + 0.3 * dst_flip); 

		dstVec.push_back(dst_c);
	}
	
	cv::Mat dstMats[] = {dstVec[0], dstVec[1], dstVec[2]};

	cv::merge(dstMats, 3, dst);

	customCV::matFilter(dst, 0, 255);

	dst.convertTo(dst, CV_8UC3); 

	imshow("src", src_1);
	imshow("dst", dst);
	imwrite("ddd.png", dst);
	imwrite(save_name, dst);
		
	cv::waitKey();
}



}