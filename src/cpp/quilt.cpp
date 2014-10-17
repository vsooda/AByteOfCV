#include "photoAlgo.h"
#define INF 100000000

#include "3rdparty/maxflow/graph.h"

typedef int capType;
typedef Graph<int, capType, int> GraphType;
#define SEAM_BONUS 5
#define PLACE_ENTM_TESTS 100

namespace customCV {
	Quilting::Params::Params(int width, int iter) {
		win_ = width;
		iter_ = iter;
	}

	class QuiltingTstImpl : public Quilting {
	public:
		QuiltingTstImpl(const Params& params) {
			setParams(params);
		}
		virtual ~QuiltingTstImpl() {}
		//需要增加对于部分参数进行修改的支持，现在需要改参数，需要对所有的参数进行修改。。
		//现在的搞法是：getParam， 再对param中的某个字段进行修改	
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		cv::Mat dpmain(cv::Mat err_sq, int border) {
			//to do	
			return err_sq;
		}

		void getPathMat(cv::Mat& mins, cv::Mat& paths, cv::Mat row_data) {
			cv::Size sz = row_data.size();
			int len = sz.width;
			int depth = sz.height;

			cv::Mat data;
			data = row_data(cv::Range::all(), cv::Range(0, len - 1)) + row_data(cv::Range::all(), cv::Range(1, len));
			len = len - 1;
			mins = cv::Mat(depth, len, CV_32F, cv::Scalar(0.0));
			paths = cv::Mat(depth, len, CV_16S, cv::Scalar(0));

			data.row(depth - 1).copyTo(mins.row(depth - 1));

			for (int i = depth - 2; i >= 0; i--) {
				cv::Mat minline = mins.row(i + 1);
				cv::Mat leftline, rightline;
				minline.copyTo(leftline);
				minline.copyTo(rightline);
				leftline.at<float>(0, 0) = INF;
				minline(cv::Range(0, 1), cv::Range(0, len - 1)).copyTo(leftline(cv::Range(0, 1), cv::Range(1, len)));
				rightline.at<float>(0, len - 1) = INF;
				minline(cv::Range(0, 1), cv::Range(1, len)).copyTo(rightline(cv::Range(0, 1), cv::Range(0, len - 1)));
				cv::Mat t1, t2, t3;
				t1 = leftline + data.row(i);
				t2 = minline + data.row(i);
				t3 = rightline + data.row(i);

				for (int j = 0; j < len; j++) {
					float min_ = INF;
					int index = 0;
					//min 3;
					if (t1.at<float>(0, j) < t2.at<float>(0, j)) {
						index = 1;
						min_ = t1.at<float>(0, j);
					}
					else {
						index = 0;
						min_ = t2.at<float>(0, j);
					}
					if (t3.at<float>(0, j) < min_) {
						index = -1;
						min_ = t3.at<float>(0, j);
					}
					mins.at<float>(i, j) = min_;
					paths.at<short>(i, j) = index;
				}

			}
		}

		void path2Mask(cv::Mat& mask, cv::Mat paths, int ind) {
			if (ind < 1) {
				ind = 1;
			}
			int start = ind + 1;
			cv::Size sz = paths.size();
			int depth = sz.height;
			cv::Mat temp(1, ind, CV_32F, cv::Scalar(1));
			//bug!! 如果ind==0的话，下面一句会出错
			temp.copyTo(mask(cv::Range(ind, ind + 1), cv::Range(0, ind)));
			for (int i = start; i < depth; i++) {
				ind = ind - paths.at<short>(i, ind);
				if (ind < 0) {
					ind = 0;
				}
				//std::cout << ind << std::endl;
				cv::Mat temp(1, ind + 1, CV_32F, cv::Scalar(1));
				temp.copyTo(mask(cv::Range(i, i + 1), cv::Range(0, ind + 1)));
			}
		}

		int maskStartLine(cv::Mat mins1, cv::Mat mins2, int border) {
			int ind;
			cv::Mat dig1 = mins1(cv::Range(0, border - 1), cv::Range(0, border - 1)).diag(0);
			cv::Mat dig2 = mins2(cv::Range(0, border - 1), cv::Range(0, border - 1)).t().diag(0);
			cv::Mat minline = dig1 + dig2;
			float minCost = INF;
			for (int i = 0; i < border - 1; i++) {
				float temp = minline.at<float>(i, 0);
				if (temp < minCost) {
					minCost = temp;
					ind = i;
				}
			}
			return ind;
		}

		cv::Mat getBlendMask(cv::Mat err_sq, int border, int i, int j, int width) {
			//j 行， i列
			int maskSize = width + border - 1;
			//默认
			cv::Mat blendMask(maskSize, maskSize, CV_32F, cv::Scalar(0));
			if (i == 0 && j == 0) {
				return blendMask;
			}
			if (j == 0) { //第一行
				cv::Mat mins, paths;
				cv::Mat data;
				data = err_sq(cv::Range::all(), cv::Range(0, border));
				getPathMat(mins, paths, data);
				int ind = 2;
				path2Mask(blendMask, paths, ind);
			}
			else if (i == 0) {
				cv::Mat mins, paths;
				cv::Mat data;
				data = err_sq(cv::Range(0, border), cv::Range::all());
				data = data.t();
				getPathMat(mins, paths, data);
				int ind = 2;
				path2Mask(blendMask, paths, ind);
				blendMask = blendMask.t();
			}
			else if (i >= 1 && j >= 1) {
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
				int ind = maskStartLine(mins1, mins2, border);
				//ind = 5;
				path2Mask(blendMask, paths1, ind);
				cv::Mat blendMask2(maskSize, maskSize, CV_32F, cv::Scalar(0));
				path2Mask(blendMask2, paths2, ind);
				blendMask2 = blendMask2.t();
				blendMask = blendMask | blendMask2;

				cv::Mat temp(ind + 1, ind + 1, CV_32F, cv::Scalar(1));
				temp.copyTo(blendMask(cv::Range(0, ind + 1), cv::Range(0, ind + 1)));

			}
			return blendMask;
		}


		//暂时不考虑src_mask, dst_mask
		void xferfun(cv::Mat texture, cv::Mat src_im_map, cv::Mat& dst, cv::Mat dst_im_map, int width, int niter) {
			cv::Mat DICT_IM_SQUARED, DICT_IM;
			int border = width / 3;
			int filter_w = 4;
			cv::Size sz = dst.size();
			cv::Size src_sz = texture.size();
			int rows = sz.height;
			int cols = sz.width;
			//smooth_filter
			cv::Mat src_mask(texture.size(), CV_8UC3, cv::Scalar(0));
			src_mask.convertTo(src_mask, CV_32FC3);
			cv::Mat dst_mask(dst.size(), CV_8U, cv::Scalar(0));

			src_im_map.copyTo(DICT_IM);

			//使用uchar的话，这里可能发生溢出。导致结果的不正确。（待确定）
			cv::pow(DICT_IM, 2, DICT_IM_SQUARED);

			cv::Mat work_im_rgb;
			cv::Mat cur_dst_im, cur_dst_im_map;
			dst.copyTo(cur_dst_im);
			dst_im_map.copyTo(cur_dst_im_map);
			cur_dst_im.copyTo(work_im_rgb);
			cv::Mat work_im_rgb2(dst.size(), CV_8U, cv::Scalar(0));
			cv::namedWindow("paiting", 1);

			int flag_j = 0, flag_i = 0;

			for (int iter = 0; iter < niter; iter++) {
				flag_j = 0;
				for (int j = 0; j < rows - 1;) {
					flag_i = 0;
					for (int i = 0; i < cols - 1;) {
						cv::Rect rect(i, j, width + border - 1, width + border - 1);
						cv::Mat tmplate, sq, sq_rgb;
						cur_dst_im_map(rect).copyTo(tmplate);
						int src_i, src_j;
						if (i == 0 && j == 0) {
							std::cout << "rand " << rand() << std::endl;
							src_i = rand() % (src_sz.height - width - border);
							src_j = rand() % (src_sz.width - width - border);
						}
						else {
							cv::Point2i pt = customCV::samplePos(tmplate, DICT_IM_SQUARED, DICT_IM);
							src_i = pt.y;
							src_j = pt.x;
						}
						cv::Rect rect2(src_j, src_i, width + border - 1, width + border - 1);
						src_im_map(rect2).copyTo(sq);
						texture(rect2).copyTo(sq_rgb);
						cv::Mat err_sq;
						cv::pow(sq - tmplate, 2, err_sq);

						//blendMask的计算还有问题, 暂时先这样
						cv::Mat blendMask = getBlendMask(err_sq, border, i, j, width);
						imshow("mask", blendMask);

						//blend_mask = rconv2(double(blend_mask),smooth_filt); % Do blending      
						cv::Mat blendMask3, oneMat3;
						cv::Mat workROI = work_im_rgb(rect);
						cv::Mat oneMat = cv::Mat::ones(width + border - 1, width + border - 1, CV_32F);
						cv::Mat oneMats[] = { oneMat, oneMat, oneMat };
						cv::merge(oneMats, 3, oneMat3);
						cv::Mat blendMats[] = { blendMask, blendMask, blendMask };
						cv::merge(blendMats, 3, blendMask3);
						//imshow("workROT", workROI);
						imshow("sq_rgb", sq_rgb);
						workROI = workROI.mul(blendMask3) + sq_rgb.mul(oneMat3 - blendMask3);

						cv::Mat mapROI = cur_dst_im_map(rect);
						mapROI = tmplate.mul(blendMask) + sq.mul(oneMat - blendMask);
						imshow("paiting", work_im_rgb);
						cv::waitKey(1);
						if (flag_i == 1){
							break;
						}
						if (i + width > cols - width - border) {
							i = cols - width - border + 1;
							flag_i = 1;
						}
						else {
							i += width;
						}
					}
					if (flag_j == 1){
						break;
					}
					if (j + width > rows - width - border) {
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
			imshow("dst1", dst);
			dst.convertTo(dst, CV_32FC3);

			imshow("dst", dst);

		}


		void apply(cv::Mat texture, cv::Mat& dst, cv::InputArray maskMat = cv::noArray()) {
			cv::Mat src_im_map;
			cvtColor(texture, src_im_map, CV_RGB2GRAY);
			imshow("src_im_map", src_im_map);
			std::vector<cv::Mat> cMatVec;
			//拆分为3通道
			cv::split(dst, cMatVec);
			cv::Mat dst_im_map;
			cMatVec[0].copyTo(dst_im_map);
			xferfun(texture, src_im_map, dst, dst_im_map, params_.win_, params_.iter_);
		}


	public:
		Params params_;
	};



	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/////graph cut 
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	struct seamNode {
		uchar flag;
		std::vector<cv::Vec3b> values;
	};

	Quilting::Params::Params(bool bsample, float reduction) {
		bSample_ = bsample;
		reduction_ = reduction;
		if (reduction < 1) {
			reduction_ = 1;
		}
	}

	class QuiltingGCImpl : public Quilting {                 //graph cut
	public:
		QuiltingGCImpl(const Params& params) {
			setParams(params);
		}
		virtual ~QuiltingGCImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		void apply(cv::Mat texture, cv::Mat& dst, cv::InputArray maskMat = cv::noArray());

	public:
		void place();
		void buildGraph();
		void init();
		void doMain();
		bool getStopStutas();
		void mask_test();
		cv::Point2i getOffset();
		std::vector<int> sample(cv::Point2i offset);
		cv::Point2i placeRandom();
		bool enoughOverlap(int x, int y, int width, int height);
		bool borderPatch(int x, int y);
		bool borderDst(int x, int y);
		void addHorizonEdge(int i, int j, int k);
		void addVerticalEdge(int i, int j, int k);
		capType graphCost(int x1, int y1, int x2, int y2);
		capType graphCost(cv::Vec3b s1, cv::Vec3b t1, cv::Vec3b s2, cv::Vec3b t2);
		void initSeamVec();
		void graphCutSeam();
		void doSeam(bool lastSource, int i, int j, int x, int y);
		void placeInit();
		cv::Point2i placeEntireMatching();
		cv::Point2i placeSubPatch();
		float calcGrad(cv::Vec3b s1, cv::Vec3b t1, cv::Vec3b s2, cv::Vec3b t2);

	private:
		cv::Mat srcPatch_;
		cv::Mat patch_;
		cv::Mat mask_;
		cv::Mat dst_;
		cv::Mat err_;
		int width_, height_; //dst 大小
		int pWidth_, pHeight_;
		GraphType * G_;
		std::vector<seamNode>seah_, seav_; //记录graphcut切分点seam node
		//pos_[0], pos_[1] dst中的位置
		//pos_[2], pos_[3] patch的width，height
		//pos_[4], pos_[5] patch中的位置。在本程序中一般都是0，0， 即使有采样，也是将patch整体替换，起始位置还是0，0，暂时留着
		std::vector<int> pos_;
		bool bSample_;
		//重叠率
		float ratio_;
		//patch 修补次数
		int cnt_;
		std::vector<GraphType::node_id> nodes_;
		bool finished_;
		int pixels_;
		int squares[256];
		int sink_, source_;
		int dstFlag; // 0 表示扩展， 1 表示已经有dst和mask
		int maskError;
		int subPatchsizeW, subPatchsizeH;
		float reduction_;

	public:
		Params params_;
	};


	//先保证这里的mask不为空
	void QuiltingGCImpl::apply(cv::Mat texture, cv::Mat& dst, cv::InputArray maskMat /* = cv::noArray() */) {
		srcPatch_ = texture;
		dst_ = dst;
		mask_ = maskMat.getMat();
		width_ = dst_.cols;
		height_ = dst_.rows;
		err_ = cv::Mat(height_, width_, CV_8U, cv::Scalar(0));
		pos_ = std::vector<int>(8, 0);
		bSample_ = params_.bSample_;
		reduction_ = params_.reduction_;
		ratio_ = 0.05;
		cnt_ = 0;
		pWidth_ = patch_.cols;
		pHeight_ = patch_.rows;
		finished_ = false;
		pixels_ = 0;
		dstFlag = 1;
		if (mask_.empty()) {
			mask_ = cv::Mat(dst.size(), CV_8U, cv::Scalar(0));
			dstFlag = 0;  //没有mask
		}

		subPatchsizeH = srcPatch_.rows / reduction_;
		subPatchsizeW = srcPatch_.cols / reduction_;
		srcPatch_.copyTo(patch_);

		mask_.convertTo(mask_, CV_32S);

		initSeamVec();
		for (int i = 0; i < 256; i++) {
			squares[i] = i * i;
		}

		doMain();
	}

	//主函数
	void QuiltingGCImpl::doMain() {
		int maxNode = patch_.rows * patch_.cols;
		int maxEdge = maxNode * 4;
		bool stop = false;
		placeInit();
		std::cout << "init ok" << std::endl;
		while (!stop) {
			//通过随机，采样等方式，决定最终的
			std::cout << "placing" << std::endl;
			place();

			//std::cout << "place ok" << std::endl;

			G_ = new GraphType(maxNode, maxEdge);

			//构图
			buildGraph();

			//std::cout << "build ok" << std::endl;

			//图割
			int flow = G_->maxflow();
			//std::cout << "flow ok" << std::endl;

			//根据graphcut结果处理mask，dst_等
			graphCutSeam();
			//std::cout << "cut ok" << std::endl;


			delete G_;
			cv::Mat t = cv::Mat(height_, width_, CV_8U, cv::Scalar(0));
			for (int i = 0; i < width_; i++) {
				for (int j = 0; j < height_; j++) {
					if (mask_.at<int>(j, i) > 0) {
						t.at<uchar>(j, i) = 255;
					}
				}
			}
			std::cout << "source: " << source_ << " sink: " << sink_ << std::endl;
			imshow("mask", t);
			imshow("dst", dst_);
			if (finished_)
				break;
			cv::waitKey(10);

			cnt_++;
		}
		std::cout << "maskError: " << maskError << std::endl;
		std::cout << "total pixels: " << pixels_ << std::endl;
		imshow("dst", dst_);
		imwrite("../data/gco.jpg", dst_);
		cv::waitKey();
	}

	void QuiltingGCImpl::placeInit() {
		if (dstFlag == 0) {
			int x, y;
			x = (int)((width_ - patch_.cols) * ((float)rand() / RAND_MAX));
			y = (int)((height_ - patch_.rows) * ((float)rand() / RAND_MAX));
			for (int i = 0; i < patch_.cols; i++) {
				for (int j = 0; j < patch_.rows; j++) {
					int xx = x + i, yy = y + j;
					dst_.at<cv::Vec3b>(yy, xx) = patch_.at<cv::Vec3b>(j, i);
					mask_.at<int>(yy, xx) = 255;
					pixels_++;
				}
			}
		}
		else if (dstFlag == 1) {
			for (int i = 0; i < width_; i++) {
				for (int j = 0; j < height_; j++) {
					if (mask_.at<int>(j, i) > 10) {
						mask_.at<int>(j, i) = 255;
						pixels_++;
					}
					else {
						mask_.at<int>(j, i) = 0;
					}
				}
			}
		}

	}

	//先取得在目标图片中的填充位置，再patch中采样，并根据边界限制，获取填充的位置，大小
	void QuiltingGCImpl::place() {

		cv::Point2i offset;
		if (bSample_) {
			offset = placeSubPatch();
		}
		else {
			offset = placeEntireMatching();
		}

		std::cout << offset << std::endl;
		pos_[0] = offset.x;
		pos_[1] = offset.y;

		std::vector<int> samplePos(4, 0);
		if (bSample_) {
			//sample函数应保证采样不越界
			samplePos = sample(offset);
		}
		else {
			samplePos[0] = 0;
			samplePos[1] = 0;
			samplePos[2] = patch_.cols;
			samplePos[3] = patch_.rows;
		}


		pos_[2] = samplePos[2];
		pos_[3] = samplePos[3];

		pos_[4] = samplePos[0];
		pos_[5] = samplePos[1];
	}


	cv::Point2i QuiltingGCImpl::placeEntireMatching() {
		int bestX = 0, bestY = 0;
		int fullArea = patch_.cols * patch_.rows;
		int bound = fullArea * ratio_;
		float cost = 100000000, ncost;
		bool bHoles = false;
		int cnt = PLACE_ENTM_TESTS;
		int ok = 0;
		int sumR, sumG, sumB;
		int x, y;
		int k = 0;
		while (cnt--) {
			ok = 0;
			while (!ok) {
				k = 0;
				sumR = sumB = sumG = 0;
				x = (int)(width_ * ((float)rand() / RAND_MAX));
				y = (int)(height_ *((float)rand() / RAND_MAX));
				for (int i = x; i < x + patch_.cols; i++) {
					for (int j = y; j < y + patch_.rows; j++) {
						if (mask_.at<int>(j%height_, i%width_)) {
							k++;
						}
					}
				}
				if (k >= bound) {
					ok = 1;
				}
				if (k == fullArea) {
					bHoles = false;
				}
				else {
					bHoles = true;
				}
			}
			if (bHoles) {
				for (int i = x; i < x + patch_.cols; i++) {
					for (int j = y; j < y + patch_.rows; j++) {
						if (mask_.at<int>(j % height_, i % width_)) {
							cv::Vec3b temp;
							temp = patch_.at<cv::Vec3b>(j - y, i - x) - dst_.at<cv::Vec3b>(j%height_, i%width_);
							sumB += squares[abs(temp[0])];
							sumG += squares[abs(temp[1])];
							sumR += squares[abs(temp[2])];
						}
					}
				}
				sumR = (int)(sumR / k);
				sumG = (int)(sumG / k);
				sumB = (int)(sumB / k);
				ncost = (sumR + sumG + sumB) / 3.0;
				if (bHoles) {
					ncost = ncost * 0.75;
				}
				if (cost > ncost) {
					cost = ncost;
					bestX = x;
					bestY = y;
				}
			}
		}
		return cv::Point2i(bestX, bestY);
	}

	//只需要保证覆盖率即可。采用用sample函数
	cv::Point2i QuiltingGCImpl::placeSubPatch() {
		int fullArea = subPatchsizeW * subPatchsizeH;
		int bound = fullArea * ratio_;
		int x, y;
		while (true) {
			int k = 0;
			x = (int)(width_ * ((float)rand() / RAND_MAX));
			y = (int)(height_ *((float)rand() / RAND_MAX));
			for (int i = x; i < x + subPatchsizeW; i++) {
				for (int j = y; j < y + subPatchsizeH; j++) {
					if (mask_.at<int>(j%height_, i%width_)) {
						k++;
					}
				}
			}
			if (k > bound && k < fullArea) {
				break;
			}
		}
		return cv::Point2i(x, y);
	}


	//保证不会越界
	std::vector<int> QuiltingGCImpl::sample(cv::Point2i offset) {
		std::vector<int> samplePos;
		samplePos = std::vector<int>(4, 0);
		//重点: 决定采样位置
		cv::Mat DICT_IM, DICT_IM_SQUARED, tmplate;
		cv::Mat temp;
		cv::Mat bigDst;
		hconcat(dst_, dst_, bigDst);
		vconcat(bigDst, bigDst, bigDst);

		cv::Rect rect(pos_[0], pos_[1], subPatchsizeW, subPatchsizeH);
		temp = bigDst(rect);
		cvtColor(temp, temp, CV_BGR2GRAY);
		temp.copyTo(tmplate);

		cvtColor(srcPatch_, DICT_IM, CV_BGR2GRAY);

		tmplate.convertTo(tmplate, CV_32F);
		DICT_IM.convertTo(DICT_IM, CV_32F);

		DICT_IM.copyTo(DICT_IM_SQUARED);
		DICT_IM_SQUARED.convertTo(DICT_IM_SQUARED, CV_32F);

		cv::pow(DICT_IM, 2, DICT_IM_SQUARED);

		cv::Point2i pt(0, 0);
		pt = customCV::samplePos(tmplate, DICT_IM_SQUARED, DICT_IM);

		samplePos[0] = pt.x;
		samplePos[1] = pt.y;
		samplePos[2] = subPatchsizeW;
		samplePos[3] = subPatchsizeH;
		cv::Rect rect1(samplePos[0], samplePos[1], samplePos[2], samplePos[3]);
		temp = srcPatch_(rect1);
		temp.copyTo(patch_);
		return samplePos;
	}

	//use old patch and new patch to build graph
	//修改变量
	void QuiltingGCImpl::buildGraph() {
		int xMin, xMax, yMin, yMax;
		xMin = pos_[0];
		xMax = pos_[0] + pos_[2];
		yMin = pos_[1];
		yMax = pos_[1] + pos_[3];
		nodes_.clear();

		int k = 0;
		int sink = 0, sources = 0, nos = 0;
		for (int i = 0; i < 6; i++) {
			std::cout << pos_[i] << " ";
		}
		std::cout << std::endl;

		for (int x = xMin; x < xMax; x++) {
			for (int y = yMin; y < yMax; y++) {
				if (mask_.at<int>(y % height_, x % width_) == 0) {
					continue;
				}
				nodes_.push_back(G_->add_node());
				//mask的值等于节点编号值+1，因为节点从0开始编号，而mask的0为默认值
				mask_.at<int>(y % height_, x % width_) = k + 1;

				try {
					if (borderDst(x, y)) {
						G_->add_tweights(nodes_[k], 0, INF);
						sink++;
					}
					else if (borderPatch(x, y)) {
						G_->add_tweights(nodes_[k], INF, 0);
						sources++;
					}
					else {
						nos++;
					}

					addHorizonEdge(x, y, k);
					addVerticalEdge(x, y, k);
				}
				catch (int &val) {
					std::cout << "error code: " << val << std::endl;
					std::cout << x << " " << y << " " << k << std::endl;
					exit(-1);
				}
				k++;
			}
		}
		std::cout << "build: source: " << sources << " sink " << sink << std::endl;
	}

	//patch边缘，当作源点 G_->set_tweights( nodes[k], MAX_SHORT, 0 );
	bool QuiltingGCImpl::borderPatch(int x, int y) {
		try {
			for (int i = -1; i < 2; i++) {
				for (int j = -1; j < 2; j++) {
					int xx = (x + i + width_) % width_;
					int yy = (y + j + height_) % height_;
					if (mask_.at<int>(yy, xx) == 0) {
						return true;
					}
				}
			}
			return false;
		}
		catch (...) {
			std::cout << "borderPatch error" << std::endl;
			throw 2;
		}
	}


	//Dst边缘 当作汇点 G->set_tweights( nodes[k], 0, MAX_SHORT );
	//这个实现跟kuav略有不同，需要注意一下
	bool QuiltingGCImpl::borderDst(int x, int y) {
		try {
			if (x == pos_[0] || y == pos_[1])
				return true;
			int xx = pos_[0] + pos_[2] - 1;
			if (xx > patch_.cols) {
				if (x == xx - 1) {
					return true;
				}
			}
			else if (x == xx) {
				return true;
			}
			int yy = pos_[1] + pos_[3] - 1;
			if (yy > patch_.rows) {
				if (y == yy - 1) {
					return true;
				}
			}
			else if (y == yy)
				return true;
			return false;
		}
		catch (...) {
			std::cout << "borderDst error " << std::endl;
			throw 1;
		}
	}

	//添加水平边
	void QuiltingGCImpl::addHorizonEdge(int x, int y, int k){
		int patchX = (x + width_ - pos_[0]) % patch_.cols;
		int patchY = (y + height_ - pos_[1]) % patch_.rows;
		int patchX_1 = (x - 1 + width_ - pos_[0]) % patch_.cols;
		int patchY_1 = (y - 1 + height_ - pos_[1]) % patch_.rows;
		x = x % width_;
		y = y % height_;
		int x_1 = (x - 1 + width_) % width_;
		int nodeIndex;
		try {
			if (mask_.at<int>(y, x_1) > 0) {
				if (seah_[x*height_ + y].flag != 0) {
					int w = x * height_ + y;

					//add its edge (to both neighbors and source
					k++;
					nodes_.push_back(G_->add_node());

					std::vector<cv::Vec3b> values;
					values = seah_[x* height_ + y].values;
					int cost = graphCost(values[0], values[1], values[2], values[3]) + SEAM_BONUS;
					G_->add_tweights(nodes_[k], cost, 0);
					nodeIndex = mask_.at<int>(y, x_1) - 1;
					cost = graphCost(values[0], patch_.at<cv::Vec3b>(patchY, patchX_1),
						values[3], patch_.at<cv::Vec3b>(patchY, patchX)
						);

					if (nodeIndex >= nodes_.size()){
						return;
					}
					if (nodes_[k] == nodes_[nodeIndex]) {
						return;
					}
					G_->add_edge(nodes_[k], nodes_[nodeIndex], cost, cost);

					//k is the index of seam node , and k-1 is the index of current pixel
					cost = graphCost(values[1], patch_.at<cv::Vec3b>(patchY, patchX_1),
						values[2], patch_.at<cv::Vec3b>(patchY, patchX)
						);

					G_->add_edge(nodes_[k], nodes_[k - 1], cost, cost);
				}
				else {
					nodeIndex = mask_.at<int>(y, x_1) - 1;
					if (nodeIndex >= nodes_.size()){
						return;
					}
					capType ncost = graphCost(x, y, x_1, y);
					if (nodes_[k] == nodes_[nodeIndex]) {
						return;
					}
					G_->add_edge(nodes_[k], nodes_[nodeIndex], ncost, ncost);
				}
			}
		}
		catch (...) {
			std::cout << "x= " << x << " y= " << y << "x_1= " << x_1 << std::endl;
			std::cout << "patchX=" << patchX << " patchY= " << patchY << " patchX_1= " << patchX_1 << std::endl;
			std::cout << "nodeINdex: " << nodeIndex << " size: " << nodes_.size() << std::endl;
			//exit(-1);
			throw 1;
		}

	}


	//添加垂直边
	void QuiltingGCImpl::addVerticalEdge(int x, int y, int k) {
		int patchX = (x + width_ - pos_[0]) % patch_.cols;
		int patchY = (y + height_ - pos_[1]) % patch_.rows;
		int patchX_1 = (x - 1 + width_ - pos_[0]) % patch_.cols;
		int patchY_1 = (y - 1 + height_ - pos_[1]) % patch_.rows;
		x = x % width_;
		y = y % height_;
		int y_1 = (y - 1 + height_) % height_;
		int nodeIndex;
		//不是边缘
		try {
			if (mask_.at<int>(y_1, x) > 0) {
				if (seav_[x*height_ + y].flag != 0) {
					int w = x * height_ + y;
					k++;
					nodes_.push_back(G_->add_node());

					std::vector<cv::Vec3b> values;
					values = seav_[x* height_ + y].values;
					int cost = graphCost(values[0], values[1], values[2], values[3]) + SEAM_BONUS;
					G_->add_tweights(nodes_[k], cost, 0);
					nodeIndex = mask_.at<int>(y_1, x) - 1;
					cost = graphCost(values[0], patch_.at<cv::Vec3b>(patchY_1, patchX),
						values[3], patch_.at<cv::Vec3b>(patchY, patchX)
						);
					if (nodeIndex >= nodes_.size()){
						return;
					}
					if (nodes_[k] == nodes_[nodeIndex]) {
						return;
					}
					G_->add_edge(nodes_[k], nodes_[nodeIndex], cost, cost);


					cost = graphCost(values[1], patch_.at<cv::Vec3b>(patchY_1, patchX),
						values[2], patch_.at<cv::Vec3b>(patchY, patchX)
						);
					G_->add_edge(nodes_[k], nodes_[k - 1], cost, cost);
				}
				else {
					nodeIndex = mask_.at<int>(y_1, x) - 1;
					if (nodeIndex >= nodes_.size()){
						return;
					}
					capType ncost = graphCost(x, y_1, x, y);
					if (nodes_[k] == nodes_[nodeIndex]) {
						return;
					}
					G_->add_edge(nodes_[k], nodes_[nodeIndex], ncost, ncost);
				}
			}
		}
		catch (...) {
			std::cout << "x= " << x << " y= " << y << "x_1= " << y_1 << std::endl;
			std::cout << "patchX=" << patchX << " patchY= " << patchY << " patchY_1= " << patchY_1 << std::endl;
			std::cout << "nodeINdex: " << nodeIndex << " size: " << nodes_.size() << std::endl;
			throw 2;
		}

	}


	//	Simplest matching quality cost function.
	//  M(s,t,A,D) = |A(s)-B(s)| + |A(t)-B(t)|
	//  ( A is the current texture, B is the patch )
	//  We assume that s and t are pixels overlapping
	capType QuiltingGCImpl::graphCost(int i, int j, int x, int y) {
		i = (i + width_) % width_;
		j = (j + height_) % height_;
		x = (x + width_) % width_;
		y = (y + height_) % height_;
		int patchI = (i + width_ - pos_[0]) % width_ % patch_.cols;
		int patchJ = (j + height_ - pos_[1]) % height_ % patch_.rows;
		int patchX = (x + width_ - pos_[0]) % width_ % patch_.cols;
		int patchY = (y + height_ - pos_[1]) % height_ % patch_.rows;

		assert(patchI >= 0 && patchI < patch_.cols);
		assert(patchX >= 0 && patchX < patch_.cols);
		assert(patchJ >= 0 && patchJ < patch_.rows);
		assert(patchY >= 0 && patchY < patch_.rows);

		cv::Vec3b a0 = dst_.at<cv::Vec3b>(j, i);
		cv::Vec3b a1 = dst_.at<cv::Vec3b>(y, x);
		cv::Vec3b b0 = patch_.at<cv::Vec3b>(patchJ, patchI);
		cv::Vec3b b1 = patch_.at<cv::Vec3b>(patchY, patchX);

		return graphCost(a0, b0, a1, b1);
	}

	capType QuiltingGCImpl::graphCost(cv::Vec3b s1, cv::Vec3b t1, cv::Vec3b s2, cv::Vec3b t2) {
		int B = abs(s1[0] - t1[0]) + abs(s2[0] - t2[0]);
		int G = abs(s1[1] - t1[1]) + abs(s2[1] - t2[1]);
		int R = abs(s1[2] - t1[2]) + abs(s2[2] - t2[2]);
		double cost = (R + G + B) / 3;
		return cost;
		//capType ret = 0;
		//ret = norm(s1, t1) + norm(s2, t2);
		//return ret;
	}

	float QuiltingGCImpl::calcGrad(cv::Vec3b s1, cv::Vec3b t1, cv::Vec3b s2, cv::Vec3b t2) {
		cv::Vec3b temp1, temp2;
		temp1 = s1 - t1;
		temp2 = s2 - t2;
		float grad = 0;
		for (int i = 0; i < 3; i++) {
			grad += abs(temp1[i]) + abs(temp2[i]);
		}
		return grad / 6;
	}

	void QuiltingGCImpl::initSeamVec() {
		for (int i = 0; i < width_ * height_; i++) {
			seamNode sn;
			sn.flag = 0;
			std::vector<cv::Vec3b> val(4);
			sn.values = val;
			seav_.push_back(sn);
			seah_.push_back(sn);
		}
	}

	//对graphcut结果进行处理，找出分界点，并标记
	void QuiltingGCImpl::graphCutSeam() {
		bool lastSource = true;
		bool firstOv = true;
		sink_ = 0; source_ = 0;
		int totalPixels = width_ * height_;
		for (int i = 0; i < patch_.cols; i++) {
			firstOv = true;
			for (int j = 0; j < patch_.rows; j++) {
				int x = (i + pos_[0]) % width_;
				int y = (j + pos_[1]) % height_;
				int nodeIndex = mask_.at<int>(y, x);
				assert(nodeIndex <= nodes_.size());
				if (nodeIndex == 0) {
					dst_.at<cv::Vec3b>(y, x) = patch_.at<cv::Vec3b>(j, i);
					mask_.at<int>(y, x) = 255;
					pixels_++;
					continue;
				}
				if (pixels_ >= totalPixels) {
					finished_ = true;
				}
				else {
					if (firstOv) {
						if (G_->what_segment(nodes_[nodeIndex - 1]) == GraphType::SOURCE) {
							lastSource = true;
						}
						else {
							lastSource = false;
						}
						firstOv = false;
					}
					try {
						doSeam(lastSource, i, j, x, y);
					}
					catch (...) {
						std::cerr << "doSeam Error " << i << " " << j << " " << x << " " << y << std::endl;
						exit(-1);
					}
				}
			}
		}
	}

	//处理graphcut, 每个点
	void QuiltingGCImpl::doSeam(bool lastSource, int i, int j, int x, int y) {
		int nodeIndex = mask_.at<int>(y, x);
		assert(nodeIndex <= nodes_.size() && nodeIndex - 1 >= 0);
		int x_1 = (x - 1 + width_) % width_;
		int y_1 = (y - 1 + height_) % height_;
		int i_1 = (i - 1 + patch_.cols) % patch_.cols;
		int j_1 = (j - 1 + patch_.rows) % patch_.rows;
		if (G_->what_segment(nodes_[nodeIndex - 1]) == GraphType::SOURCE) {
			source_++;
			dst_.at<cv::Vec3b>(y, x) = patch_.at<cv::Vec3b>(j, i);

			// if last pixel was from SINK, draw a black pixel, o/w draw white
			if (!lastSource) {
				err_.at<uchar>(y, x) = 0;
				lastSource = true;
				//add seam node
				int w = x * height_ + y;
				seav_[w].flag = 1;
				seav_[w].values[0] = dst_.at<cv::Vec3b>(y_1, x);
				seav_[w].values[1] = patch_.at<cv::Vec3b>(j_1, i);
				seav_[w].values[2] = dst_.at<cv::Vec3b>(y, x);
				seav_[w].values[3] = patch_.at<cv::Vec3b>(j, i);

				// Look at THE LEFT PIXEL: IF different origin, THEN add seam node 
				int xx = (i - 1 + pos_[0]) % width_;
				int yy = (j + pos_[1]) % height_;
				int nodex = mask_.at<int>(yy, xx);
				int aa = nodes_.size();
				//在kuv程序中没有这种处理，这很可能是导致其程序崩溃的主要原因
				//patch较小的时候更有可能发生这种情况，原因是？
				if ((nodex == 255 && nodex > nodes_.size()) || nodex == 0) {
					maskError++;
					std::cout << "nodex error " << nodex << std::endl;
					return;
				}
				if (G_->what_segment(nodes_[mask_.at<int>(yy, xx) - 1]) == GraphType::SINK) {
					int w = x * height_ + y;
					seah_[w].flag = 1;
					seah_[w].values[0] = dst_.at<cv::Vec3b>(y, x_1);
					seah_[w].values[1] = patch_.at<cv::Vec3b>(j, i_1);
					seah_[w].values[2] = dst_.at<cv::Vec3b>(y, x);
					seah_[w].values[3] = patch_.at<cv::Vec3b>(j, i);
				}
			}
			else {
				err_.at<uchar>(y, x) = 255;
			}
		}
		else if (G_->what_segment(nodes_[nodeIndex - 1]) == GraphType::SINK) {
			sink_++;
			if (lastSource) {
				err_.at<uchar>(y, x) = 0;
				lastSource = false;

				int w = x * height_ + y;
				seav_[w].flag = 1;
				seav_[w].values[0] = patch_.at<cv::Vec3b>(j_1, i);
				seav_[w].values[1] = dst_.at<cv::Vec3b>(y_1, x);
				seav_[w].values[2] = patch_.at<cv::Vec3b>(j, i);
				seav_[w].values[3] = dst_.at<cv::Vec3b>(y, x);

				int xx = (i - 1 + pos_[0]) % width_;
				int yy = (j + pos_[1]) % height_;
				int nodex = mask_.at<int>(yy, xx);
				int aa = nodes_.size();
				if ((nodex == 255 && nodex > nodes_.size()) || nodex == 0) {
					maskError++;
					std::cout << "nodex error " << nodex << std::endl;
					return;
				}
				if (G_->what_segment(nodes_[mask_.at<int>(yy, xx) - 1]) == GraphType::SOURCE) {
					seah_[w].flag = 1;
					seah_[w].values[0] = patch_.at<cv::Vec3b>(j, i_1);
					seah_[w].values[1] = dst_.at<cv::Vec3b>(y, x_1);
					seah_[w].values[2] = patch_.at<cv::Vec3b>(j, i);
					seah_[w].values[3] = dst_.at<cv::Vec3b>(y, x);
				}
			}
		}
	}



/////////////////////////////////////////////////////////////////////////////




	cv::Ptr<Quilting> Quilting::create(int type, const Params& params) {
		if (type == QUILT_TST) {
			return cv::Ptr<Quilting>(new QuiltingTstImpl(params));
		}
		else if (type == QUILT_GRAPHCUT) {
			return cv::Ptr<Quilting>(new QuiltingGCImpl(params));
		}
	}
}