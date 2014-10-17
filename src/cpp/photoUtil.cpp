#include "photoUtil.h"
namespace customCV {

	cv::Mat Erosion(cv::Mat src, int size, int type)
	{
		cv::Mat dst;
		int erosion_type;
		int erosion_size = size;
		if (type == 0){ erosion_type = cv::MORPH_RECT; }
		else if (type == 1){ erosion_type = cv::MORPH_CROSS; }
		else if (type == 2) { erosion_type = cv::MORPH_ELLIPSE; }

		cv::Mat element = cv::getStructuringElement(erosion_type,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		erode(src, dst, element);
		return dst;
	}

	cv::Mat  Dilation(cv::Mat src, int size, int type)
	{
		cv::Mat dst;
		int dilation_type;
		int dilation_size = size;
		if (type == 0){ dilation_type = cv::MORPH_RECT; }
		else if (type == 1){ dilation_type = cv::MORPH_CROSS; }
		else if (type == 2) { dilation_type = cv::MORPH_ELLIPSE; }

		cv::Mat element = cv::getStructuringElement(dilation_type,
			cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			cv::Point(dilation_size, dilation_size));
		dilate(src, dst, element);
		return dst;
	}


	//处理uchar类型
	void  transformMat(cv::Mat& src, cv::Scalar avg_src, cv::Scalar avg_dst, cv::Scalar std_src, cv::Scalar std_dst, cv::InputArray maskMat) {
		cv::Mat mask = maskMat.getMat();
		int rows = src.rows;
		int cols = src.cols;
		for (int i = 0; i < 3; i++) {
			for (int x = 0; x < rows; x++) {
				uchar *ptr = (uchar*)(src.data + x * src.step);
				uchar *pmask = NULL;
				if (!mask.empty()) {
					pmask = (uchar*)(mask.data + x * mask.step);
				}
				for (int y = 0; y < cols; y++) {
					if (!mask.empty())
					if (pmask[y] < 5)
						continue;
					double tmp = ptr[3 * y + i];
					int t = (int)((tmp - avg_dst.val[i]) * (std_src.val[i] / std_dst.val[i]) + avg_src.val[i]);
					t = t < 0 ? 0 : t;
					t = t > 255 ? 255 : t;
					ptr[3 * y + i] = t;
				}
			}
		}
	}


	//tmplate 是目标位置的灰度图
	//DICT_IM 是采样patch的灰度图，而DICT_IM_SQUARED是其平方。
	//数据类型均为float
	cv::Point2i samplePos(cv::Mat tmplate, cv::Mat DICT_IM_SQUARED, cv::Mat DICT_IM) {
		cv::Point2i pt;
		pt.x = 0;
		pt.y = 0;
		float errThreshold = 0.1;
		//template 是单通道的，可以直接搞。
		cv::Size sz = tmplate.size();
		//get the mask
		cv::Mat mask(sz.height, sz.width, CV_16S, cv::Scalar(0));
		int count = 0;
		for (int i = 0; i < sz.height; i++) {
			for (int j = 0; j < sz.width; j++) {
				if (tmplate.at<float>(i, j) > 0) {
					mask.at<short>(i, j) = 1;
					count++;
				}
			}
		}

		cv::Mat flipMask;
		cv::flip(mask, flipMask, -1);
		cv::Mat flipTemp;
		cv::flip(tmplate, flipTemp, -1);
		cv::Mat add1, add2;
		cv::Mat temp;
		cv::filter2D(DICT_IM_SQUARED, add1, -1, flipMask);

		cv::Mat floatMask;
		mask.convertTo(floatMask, CV_32F);

		temp = (tmplate.mul(tmplate));
		cv::filter2D(DICT_IM, add2, -1, flipTemp);
		add2 = add2 * (-2.0);
		cv::Scalar s = cv::sum(temp);
		float buf[4];
		cv::scalarToRawData(s, buf, tmplate.type(), 0);
		cv::Mat disMat = add1 + add2 + buf[0];
		cv::sqrt(disMat, disMat);
		double n;
		cv::minMaxLoc(disMat, 0, &n, 0, 0);
		disMat = disMat / n;

		//对应于matlab的conv2的valid参数
		disMat = disMat.colRange((mask.cols - 1) / 2, disMat.cols - mask.cols / 2)
			.rowRange((mask.rows - 1) / 2, disMat.rows - mask.rows / 2);
		double min_;
		cv::minMaxLoc(disMat, &min_, 0, &pt, 0);
		std::cout << min_ << std::endl;
		double max_ = (1 + errThreshold) * min_;
		std::vector<cv::Point2i> vec;
		int rows = disMat.rows;
		int cols = disMat.cols;
		for (int j = 0; j <= rows - 1; j++) {
			for (int i = 0; i <= cols - 1; i++) {
				if (disMat.at<float>(j, i) < max_) {
					vec.push_back(cv::Point2i(i, j));
				}
			}
		}

		int randIndex = rand() % vec.size();
		pt = vec[randIndex];

		return pt;
	}

	//使用log函数来计算肤色美白，效果不错
	void  whiteSkin(const cv::Mat &src, cv::Mat& dst, int beta) {
		assert(beta >= 2 && beta <= 5);
		cv::Mat srcCopy;
		src.convertTo(srcCopy, CV_32FC3);
		cvConvertScale(&((IplImage)srcCopy), &((IplImage)srcCopy), 1.0 / 255.0);
		srcCopy.copyTo(dst);
		float div = log(beta);
		for (int i = 0; i < srcCopy.cols; i++) {
			for (int j = 0; j < srcCopy.rows; j++) {
				for (int c = 0; c< 3; c++) {
					float temp = srcCopy.at<cv::Vec3f>(j, i)[c];
					dst.at<cv::Vec3f>(j, i)[c] = log(temp * (beta - 1) + 1) / div;
				}
			}
		}
		cvConvertScale(&((IplImage)dst), &((IplImage)dst), 255);
		dst.convertTo(dst, CV_8UC3);
	}

	void whiteSkinC1(const cv::Mat& src, cv::Mat& dst, int beta) {
		assert(beta >= 2 && beta <= 5);
		cv::Mat srcCopy;
		src.convertTo(srcCopy, CV_32F);
		cvConvertScale(&((IplImage)srcCopy), &((IplImage)srcCopy), 1.0 / 255.0);
		srcCopy.copyTo(dst);
		float div = log(beta);
		for (int i = 0; i < srcCopy.cols; i++) {
			for (int j = 0; j < srcCopy.rows; j++) {
				float temp = srcCopy.at<float>(j, i);
				dst.at<float>(j, i) = log(temp * (beta - 1) + 1) / div;
			}
		}
		dst.convertTo(dst, CV_8U, 255);
	}

	cv::Mat colorTransform(cv::Mat src, cv::Mat dst,  cv::InputArray maskMat)
	{
		int width = src.cols;
		int height = src.rows;
		cv::Mat mask = maskMat.getMat();
		if (mask.empty()) {
			mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
		}

		cv::Scalar avg_src, avg_dst, std_src, std_dst;
		meanStdDev(src, avg_src, std_src, mask);
		meanStdDev(dst, avg_dst, std_dst, mask);

		cv::Mat dstlab;
		dst.copyTo(dstlab);
		transformMat(dstlab, avg_src, avg_dst, std_src, std_dst, mask);

		return dstlab;
	}




}
