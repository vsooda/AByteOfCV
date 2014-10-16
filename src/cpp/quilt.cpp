#include "quilt.h"
#define INF 100000000
namespace customCV {

	cv::Mat dpmain(cv::Mat err_sq, int border) {
		//to do	
		return err_sq;
	}

	void getPathMat(cv::Mat& mins, cv::Mat& paths, cv::Mat row_data) {
		cv::Size sz = row_data.size();
		int len = sz.width;
		int depth = sz.height;
		
		cv::Mat data;
		data = row_data(cv::Range::all(), cv::Range(0, len-1)) + row_data(cv::Range::all(), cv::Range(1, len));
		len = len - 1;
		mins = cv::Mat(depth, len, CV_32F, cv::Scalar(0.0));
		paths = cv::Mat(depth, len, CV_16S, cv::Scalar(0));

		data.row(depth-1).copyTo(mins.row(depth-1));

		for(int i = depth-2; i >=0; i--) {
			cv::Mat minline = mins.row(i+1);
			/*	
			//这种方式效率过低，用矩阵会比较快
			for(int j = 0; j < len; j++) {
				float min_ = INF;
				if(j - 1 >= 0) { 
					float temp = data.at<float>(i,j) + mins.at<float>(i+1, j-1);
					if(temp < min_) {
						mins.at<float>(i, j) = temp;
						paths.at<short>(i, j) = 1; 
						min_ = temp;
					}
				}
				if(j + 1 < len) {
					float temp = data.at<float>(i,j) + mins.at<float>(i+1, j+1);
					if(temp < min_) {
						mins.at<float>(i, j) = temp;
						paths.at<short>(i, j) = -1; 
						min_ = temp;
					}	
				}
				{
					float temp = data.at<float>(i,j) + mins.at<float>(i+1, j);
					if(temp < min_) {
						mins.at<float>(i, j) = temp;
						paths.at<short>(i, j) = 0; 
					}	
				}
			} */
			
			cv::Mat leftline, rightline;
			minline.copyTo(leftline);
			minline.copyTo(rightline);
			leftline.at<float>(0, 0) = INF;
			minline(cv::Range(0, 1), cv::Range(0, len-1)).copyTo(leftline(cv::Range(0,1), cv::Range(1, len)));
			rightline.at<float>(0, len-1) = INF;
			minline(cv::Range(0, 1), cv::Range(1, len)).copyTo(rightline(cv::Range(0,1), cv::Range(0, len-1)));
			cv::Mat t1, t2, t3;
			t1 = leftline + data.row(i);
			t2 = minline + data.row(i);
			t3 = rightline + data.row(i);

			for(int j = 0; j < len; j++) {
				float min_ = INF;
				int index = 0;
				//min 3;
				if(t1.at<float>(0, j) < t2.at<float>(0, j)) {
					index = 1;
					min_ = t1.at<float>(0, j);
				}
				else {
					index = 0;
					min_ = t2.at<float>(0, j);
				}
				if(t3.at<float>(0, j) < min_) {
					index = -1;
					min_ = t3.at<float>(0, j);
				}
				mins.at<float>(i, j) = min_;
				paths.at<short>(i, j) = index;
			}
			
		} 
	}

	void path2Mask(cv::Mat& mask, cv::Mat paths, int ind) {
		if(ind < 1) {
			ind = 1;
		}
		int start = ind + 1;
		cv::Size sz = paths.size();
		int depth = sz.height; 
		cv::Mat temp(1 , ind, CV_32F, cv::Scalar(1));
		//bug!! 如果ind==0的话，下面一句会出错
		temp.copyTo(mask(cv::Range(ind, ind+1), cv::Range(0, ind)));
		for(int i = start; i < depth; i++) {
			ind = ind - paths.at<short>(i, ind);
			if(ind < 0) {
				ind  = 0;
			}
			//std::cout << ind << std::endl;
			cv::Mat temp(1 , ind+1, CV_32F, cv::Scalar(1));
			temp.copyTo(mask(cv::Range(i, i+1), cv::Range(0, ind+1)));
		}
	}

	int maskStartLine(cv::Mat mins1, cv::Mat mins2, int border) {
		int ind;
		cv::Mat dig1 = mins1(cv::Range(0, border-1), cv::Range(0, border-1)).diag(0);
		cv::Mat dig2 = mins2(cv::Range(0, border-1), cv::Range(0, border-1)).t().diag(0);
		cv::Mat minline = dig1 + dig2;
		float minCost = INF;
		for(int i = 0; i < border - 1; i++) {
			float temp = minline.at<float>(i, 0);
			if(temp < minCost) {
				minCost = temp;
				ind = i;	
			}
		}
		return ind;
	}

	cv::Mat getBlendMask(cv::Mat err_sq, int border, int i, int j, int width) {
		//j 行， i列
		int maskSize = width + border -1;
		//默认
		cv::Mat blendMask(maskSize , maskSize, CV_32F, cv::Scalar(0));
		if(i == 0 && j == 0) {
			return blendMask; 
		}
		if(j == 0) { //第一行
			cv::Mat mins, paths;
			cv::Mat data;
			data = err_sq(cv::Range::all(), cv::Range(0, border));
			getPathMat(mins, paths, data);
			int ind = 2;
			path2Mask(blendMask, paths, ind);	
		}
		else if(i == 0) {
			cv::Mat mins, paths;
			cv::Mat data;
			data = err_sq(cv::Range(0, border), cv::Range::all());
			data = data.t();
			getPathMat(mins, paths, data);
			int ind = 2;
			path2Mask(blendMask, paths, ind);		
			blendMask = blendMask.t();
		}
		else if(i >= 1 && j >= 1) {
			//dp1 get the path 
			cv::Mat mins1, paths1, mins2, paths2;
			//data is the some of err_sq;
			cv::Mat data1, data2;
			data1 = err_sq(cv::Range::all(), cv::Range(0, border));
			data2 = err_sq(cv::Range(0, border), cv::Range::all());
			//转置
			data2 = data2.t();
			
			getPathMat(mins1, paths1, data1);
			getPathMat(mins2, paths2, data2);
			//std::cout << paths1 << std::endl;
			//std::cout << paths2 << std::endl;
			int ind = maskStartLine(mins1, mins2, border);
			//ind = 5;
			path2Mask(blendMask, paths1, ind);
			cv::Mat blendMask2(maskSize , maskSize, CV_32F, cv::Scalar(0));
			path2Mask(blendMask2, paths2, ind);
			blendMask2 = blendMask2.t();
			blendMask = blendMask | blendMask2;

			cv::Mat temp(ind+1 , ind+1, CV_32F, cv::Scalar(1));
			temp.copyTo(blendMask(cv::Range(0, ind+1), cv::Range(0, ind+1)));

		}	
		//std::cout << blendMask << std::endl;  //mat 可以输出
		return blendMask;
	}

	cv::Point2i samplePos(cv::Mat tmplate, cv::Mat DICT_IM_SQUARED, cv::Mat DICT_IM) {
		cv::Point2i pt; 
		pt.x = 50;
		pt.y = 50;
		float errThreshold = 0.1;
		//template 是单通道的，可以直接搞。
		//std::cout << tmplate << std::endl;
		cv::Size sz = tmplate.size();
		//get the mask
		cv::Mat mask(sz.width, sz.height, CV_16S, cv::Scalar(0));
		int count = 0;
		for(int i = 0; i < sz.height; i++) {
			for(int j = 0; j < sz.width; j++) {
				if(tmplate.at<float>(i, j) > 0) {
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
		//temp = (tmplate.mul(tmplate)).mul(floatMask);

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
		disMat = disMat.colRange((mask.cols-1)/2, disMat.cols - mask.cols/2)
               .rowRange((mask.rows-1)/2, disMat.rows - mask.rows/2);

		double min_; 
		cv::minMaxLoc(disMat, &min_, 0, &pt, 0);
		double max_ = (1+errThreshold) * min_;
		std::vector<cv::Point2i> vec;
		
		int rows = disMat.rows;
		int cols = disMat.cols;
		for(int j = 0; j <  rows -1; j++) {
				for(int i = 0; i < cols - 1; i++) {
					if(disMat.at<float>(j, i) < max_) {
						vec.push_back(cv::Point2i(i, j));
					}
				}
		}

		int randIndex = rand() % vec.size();

		pt = vec[randIndex];

		std::cout << pt << std::endl;


		return pt;
	}


 

	//暂时不考虑src_mask, dst_mask
	void xferfun(cv::Mat texture, cv::Mat src_im_map, cv::Mat& dst, cv::Mat dst_im_map, int width, int niter) {
		cv::Mat DICT_IM_SQUARED, DICT_IM;
		int border = width / 3;
		int filter_w  = 4;
		cv::Size sz = dst.size();
		cv::Size src_sz = texture.size();
		int rows = sz.height;
		int cols = sz.width;
		//smooth_filter
		cv::Mat src_mask(texture.size(), CV_8U, cv::Scalar(0));
		cv::Mat dst_mask(dst.size(), CV_8U, cv::Scalar(0));

		src_im_map.copyTo(DICT_IM);
		cv::pow(DICT_IM, 2, DICT_IM_SQUARED); 

		cv::Mat work_im_rgb;
		cv::Mat cur_dst_im, cur_dst_im_map;
		dst.copyTo(cur_dst_im);
		dst_im_map.copyTo(cur_dst_im_map);
		cur_dst_im.copyTo(work_im_rgb);
		cv::Mat work_im_rgb2(dst.size(), CV_8U, cv::Scalar(0));
		cv::namedWindow("paiting",1);

		int flag_j = 0, flag_i = 0;

		for(int iter = 0; iter < niter; iter++) {
			flag_j = 0;
			for(int j = 0; j <  rows -1; ) {
				flag_i = 0;
				for(int i = 0; i < cols - 1; ) {
					cv::Rect rect(i, j, width+border-1, width+border-1);
					cv::Mat tmplate, sq, sq_rgb;
					cur_dst_im_map(rect).copyTo(tmplate);
					int src_i, src_j;
					if(i == 0 && j == 0) {
						std::cout << "rand " <<  rand() << std::endl;
						src_i = rand() % (src_sz.height- width - border);
						src_j = rand() % (src_sz.width - width - border);	
					}
					else {
						cv::Point2i pt = samplePos(tmplate, DICT_IM_SQUARED, DICT_IM);
						src_i = pt.y;
						src_j = pt.x;
					}
					cv::Rect rect2(src_j, src_i, width+border-1, width+border-1);
					src_im_map(rect2).copyTo(sq);
					texture(rect2).copyTo(sq_rgb);	
					cv::Mat err_sq;
					cv::pow(sq-tmplate, 2,	err_sq);

					//blendMask的计算还有问题, 暂时先这样
					cv::Mat blendMask = getBlendMask(err_sq, border, i, j, width);
					imshow("mask", blendMask);

					//blend_mask = rconv2(double(blend_mask),smooth_filt); % Do blending      
					cv::Mat blendMask3, oneMat3;
					cv::Mat workROI = work_im_rgb(rect);
					cv::Mat oneMat = cv::Mat::ones(width+border-1, width+border-1, CV_32F);
					cv::Mat oneMats[] = {oneMat, oneMat, oneMat};
					cv::merge(oneMats, 3, oneMat3);
					cv::Mat blendMats[] = {blendMask, blendMask, blendMask};
					cv::merge(blendMats, 3, blendMask3);
					//imshow("workROT", workROI);
					imshow("sq_rgb", sq_rgb);
					workROI =  workROI.mul(blendMask3) + sq_rgb.mul(oneMat3-blendMask3);

					cv::Mat mapROI = cur_dst_im_map(rect);
					mapROI = tmplate.mul(blendMask) + sq.mul(oneMat-blendMask);
					imshow("paiting", work_im_rgb);
					cv::waitKey(1);
					if(flag_i == 1){
						break;
					}
					if( i + width > cols - width - border) {
						i = cols - width - border + 1;
						flag_i = 1;
					}
					else {
						i += width;
					}
				}
				if(flag_j == 1){
					break;
				}
				if( j + width > rows - width - border) {
					j = rows - width - border + 1;
					flag_j = 1;
				}
				else {
					j += width;
				}
			}
			width = width * 0.8;
		}
		dst = work_im_rgb;
	}


	void quilt(cv::Mat& texture, cv::Mat& dst, int width, int niter) {
		//to do;
		cv::Mat src_im_map;
		cvtColor(texture, src_im_map, CV_RGB2GRAY); 
		imshow("src_im_map", src_im_map);
		std::vector<cv::Mat> cMatVec;
		//拆分为3通道
		cv::split(dst, cMatVec);

		cv::Mat dst_im_map;
		cMatVec[0].copyTo(dst_im_map);

		xferfun(texture, src_im_map, dst, dst_im_map, width, niter);
		//cvConvertScale(&((IplImage)dst), &((IplImage)dst), 255.0);
		//dst.convertTo(dst, CV_8UC3);
		//imwrite("dst_p6.jpg", dst);
	}











}


