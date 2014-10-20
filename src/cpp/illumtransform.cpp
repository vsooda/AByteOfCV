#include "photoAlgo.h"
#include "photoUtil.h"

namespace customCV {

	int IllumTransform::algorithmType = 0;

	IllumTransform::Params::Params(int radius, float symfactor, float absfactor, float laplacefactor) {
		radius_ = radius;
		symfactor_ = symfactor;
		absfactor_ = absfactor;
		laplacefactor_ = laplacefactor;
	}

	class IllumTransformMedianImpl : public IllumTransform {
	public:
		IllumTransformMedianImpl(const Params& params) {
			setParams(params);
		}
		virtual ~IllumTransformMedianImpl() {}
		//需要增加对于部分参数进行修改的支持，现在需要改参数，需要对所有的参数进行修改。。
		//现在的搞法是：getParam， 再对param中的某个字段进行修改	
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		virtual cv::Mat apply(cv::Mat src, cv::Mat avg, cv::InputArray maskMat = cv::noArray(), bool makeup = false) {
			//如果没有mask，制造mask
			cv::Mat mask = maskMat.getMat();
			if (mask.empty()) {
				mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255));
			}

			avg = colorTransform(src, avg, mask);

			cv::Mat dst;
			cv::Mat mask1f;
			cv::Mat src_gray;
			cvtColor(src, src_gray, CV_BGR2GRAY);
			std::vector<cv::Mat> MatVec1, MatVec2;
			std::vector<cv::Mat> dstVec, dstSymVec;
			cv::split(src, MatVec1);
			cv::split(avg, MatVec2);
			mask.convertTo(mask1f, CV_32F, 1.0 / 255);
			cv::Mat oneMat = cv::Mat::ones(src.size(), CV_32F);
			cv::Mat laplacian_w;

			CV_Assert(src_gray.type() == CV_8U);

			Laplacian(src_gray, laplacian_w, CV_16S);
			convertScaleAbs(laplacian_w, laplacian_w);

			laplacian_w.convertTo(laplacian_w, CV_32F);

			normalize(laplacian_w, laplacian_w, 0, 1.0, cv::NORM_MINMAX);

			for (int c = 0; c < 3; c++) {
				cv::Mat src_1c = MatVec1[c];
				cv::Mat src_2c = MatVec2[c];
				src_1c.convertTo(src_1c, CV_32F);
				src_2c.convertTo(src_2c, CV_32F);
				cv::Mat med1, med2;
				src_1c.copyTo(med1);
				src_2c.copyTo(med2);

				//cvSmooth(&((IplImage)src_1c), &((IplImage)med1), CV_BLUR, params_.radius_);
				//cvSmooth(&((IplImage)src_2c), &((IplImage)med2), CV_BLUR, params_.radius_);
                cv::Size sz(params_.radius_, params_.radius_);
                cv::blur(src_1c, med1, sz);
                cv::blur(src_2c, med2, sz);

				cv::Mat sym, med_gray, med_flip;
				cv::Mat sym_w;
				med1.copyTo(med_gray);
				flip(med_gray, med_flip, 1);

				sym = Symmetry(med_gray, med_flip);
				sym.convertTo(sym, CV_32F);
				normalize(sym, sym_w, 0, 1.0, cv::NORM_MINMAX);

				sym.convertTo(sym, CV_8U);

				med1.convertTo(med1, CV_32F);
				med2.convertTo(med2, CV_32F);
				cv::Mat diff = med1 - med2;

				cv::Mat w = sym_w * params_.symfactor_ + params_.absfactor_ + (1.0 - laplacian_w)* params_.laplacefactor_;
				cv::Mat diff_weight = diff.mul(w);
				cv::Mat dst_c = src_1c - diff;
				cv::Mat dst_sym_c = src_1c - diff_weight;

				dstVec.push_back(dst_c);
				dstSymVec.push_back(dst_sym_c);

			}

			cv::Mat dst_sal, dst_sym;
			cv::merge(dstVec, dst);
			cv::merge(dstSymVec, dst_sym);

			for (int i = 0; i < dst_sym.cols; i++) {
				for (int j = 0; j < dst_sym.rows; j++) {
					cv::Vec3f tempDst = dst_sym.at<cv::Vec3f>(j, i);
					for (int c = 0; c < 3; c++) {
						if (tempDst[c] > 255) {
							dst_sym.at<cv::Vec3f>(j, i)[c] = 255;
						}
						else if (tempDst[c] < 0) {
							dst_sym.at<cv::Vec3f>(j, i)[c] = 0;
						}
					}
				}
			}

			dst.convertTo(dst, CV_8UC3);
			dst_sym.convertTo(dst_sym, CV_8UC3);

			return dst_sym;
			//return dst; //未加对称性
		}

		cv::Mat Symmetry(cv::Mat lhs, cv::Mat rhs) {
			cv::Mat ret;
			cv::Mat t1, t2;
			lhs.convertTo(t1, CV_32F);
			rhs.convertTo(t2, CV_32F);
			ret = t1 - t2;
			convertScaleAbs(ret, ret);
			ret.convertTo(ret, lhs.type());
			return ret;
		}

	public:
		Params params_;
		cv::Mat src_, avg_, mask_;
	};

	IllumTransform::Params::Params(int radius, float eps,
		cv::InputArray delta, cv::InputArray mask,
		cv::InputArray transMask)  {
		radius_ = radius;
		eps_ = eps;
		delta_ = delta.getMat();
		mask_ = mask.getMat();
		transMask_ = transMask.getMat();
	}

	class IllumTransformGuideImpl : public IllumTransform {
	public:
		IllumTransformGuideImpl(const Params& params) {
			setParams(params);
		}
		virtual ~IllumTransformGuideImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		void setEpdAlgorithm(cv::Ptr<Epdfilter> epd) {
			epdMethod_ = epd;
		}

		void init() {
			if (!params_.delta_.empty()) {
				resize(params_.delta_, params_.delta_, src_.size());
			}
			else {
				params_.delta_ = cv::Mat(src_.size(), CV_8U, cv::Scalar(255));
			}
			if (!params_.mask_.empty()) {
				resize(params_.mask_, params_.mask_, src_.size());
			}
			else {
				params_.mask_ = cv::Mat(src_.size(), CV_8U, cv::Scalar(255));
			}
			if (!params_.transMask_.empty()) {
				resize(params_.transMask_, params_.transMask_, src_.size());
			}
		}

		void setEpsMat(cv::Mat delta, float factor) {
			delta = customCV::Dilation(delta, 15, 1);
			delta.convertTo(delta, CV_32F, 1 / 255.0);
			delta = 1.0 - delta;
			delta.copyTo(params_.epsMat_);
			params_.epsMat_ = 0.0001 + delta * factor;
		}

		void setSalacyDelta() {
			cv::Mat img;
			src_.convertTo(img, CV_32FC3, 1.0 / 255.0);
			cv::Ptr<customCV::Salancy> salancyMethod = customCV::Salancy::create(1);
			cv::Mat dst;
			dst = salancyMethod->apply(img, params_.mask_, p_);
			dst.convertTo(dst, CV_8UC1, 255);
			salancyDelta_ = dst;
		}

		void doMakeup(const cv::Mat& I_illum, const cv::Mat& p_illum,
						std::vector<cv::Mat>& I_vector, 
						std::vector<cv::Mat>& p_vector) {
			cv::Mat I_illum_norm = I_illum / 255.0;
			cv::Mat p_illum_norm = p_illum / 255.0;
			
			cv::Mat mask = params_.mask_;
			
			setEpsMat(params_.delta_, 0.1);

			cv::Mat q1, q2, q3;
			customCV::Epdfilter::Params params(params_.epsMat_, params_.radius_);
			if (algorithmType == GUIDE_GUIDE_FILTER) {
				setEpdAlgorithm(Epdfilter::create(Epdfilter::GUIDE_FILTER, params));
			}

			q1 = epdMethod_->apply(I_illum_norm, I_illum_norm);
			q2 = epdMethod_->apply(p_illum_norm, p_illum_norm);

			q1 = q1 * 255.0;
			q2 = q2 * 255.0;

			cv::Mat detail1, detail2;
			detail1 = I_illum - q1;
			detail2 = p_illum - q2;

			cv::Mat illum = q1 + detail2 * 0.2 + detail1 * 0.8;
			illum.convertTo(illum, I_vector[2].type());
			I_vector[0] = illum;

			I_vector[1].convertTo(I_vector[1], CV_32F);
			I_vector[2].convertTo(I_vector[2], CV_32F);
			p_vector[1].convertTo(p_vector[1], CV_32F);
			p_vector[2].convertTo(p_vector[2], CV_32F);
			mask.convertTo(mask, CV_32F, 1 / 255.0);
			I_vector[1] = I_vector[1].mul(1.0 - mask) + p_vector[1].mul(mask);
			I_vector[2] = I_vector[2].mul(1.0 - mask) + p_vector[2].mul(mask);

			I_vector[1].convertTo(I_vector[1], CV_8U);
			I_vector[2].convertTo(I_vector[2], CV_8U);

		}

		void doIllumTransform(const cv::Mat& I_illum, const cv::Mat& p_illum, 
								std::vector<cv::Mat>& I_vector, 
								std::vector<cv::Mat>& p_vector) {
			double minValue, maxValue;
			cv::Mat I_illum_norm = I_illum / 255.0;
			cv::Mat p_illum_norm = p_illum / 255.0;
			cv::Mat mask = params_.mask_;
			//params_.eps_ = 0.1;  //这个取值对细节保存结果影响很大。 取0.5会使得细节基本保留，很难消除明暗边界。取0.001则会使得细节丢失严重
			double epsDeteail = 0.05;
			setEpsMat(params_.delta_, epsDeteail);
			params_.epsMat_ = epsDeteail - params_.epsMat_ + 0.01;
			cv::Mat q1, q2, q3;
			customCV::Epdfilter::Params params(params_.epsMat_, params_.radius_);
			if (algorithmType == GUIDE_GUIDE_FILTER) {
				setEpdAlgorithm(Epdfilter::create(Epdfilter::GUIDE_FILTER, params));
			}

			q1 = epdMethod_->apply(I_illum_norm, I_illum_norm);
			q1 = q1 * 255.0;

			cv::Mat detail1, detail2;
			detail1 = I_illum - q1;
			cv::Mat detailShow;
			detailShow = abs(detail1);
			double aa, bb;
			cv::minMaxLoc(detail1, &aa, &bb);
			detail1.convertTo(detailShow, CV_32F, 1.0 / bb);

			setSalacyDelta();
			cv::imshow("saldelta", salancyDelta_);

			setEpsMat(params_.delta_, 0.1);
			cv::minMaxLoc(params_.epsMat_, &minValue, &maxValue);
			std::cout << "epsMat: " << minValue << " " << maxValue << std::endl;
			//params_.epsMat_ = 0.1;
			params_.radius_ = 10;
			//epdMethod_->setParams(params);
			q3 = epdMethod_->apply(I_illum_norm, p_illum_norm);
			//imshow("q3", q3);
			q3 = q3 * 255.0;

			//先根据光照变化情况来得到光照问题所在区域，从而对光照，颜色进行针对性处理
			cv::Mat illum;
			illum = q3 + detail1;
			illum.convertTo(illum, I_vector[2].type());
			cv::Mat illumDiff = abs(illum - I_vector[0]);
			
			cv::minMaxLoc(illumDiff, &minValue, &maxValue);
			std::cout << minValue << " " << maxValue << std::endl;
			//对光照变化较大的区域进行颜色传输

			cv::Mat diffMask;
			illumDiff.convertTo(diffMask, CV_32F, 1.0 /255.0);
			imshow("diffMask", diffMask);

			//补充空洞纹理。
			{
				cv::Mat illumtemp;
				cv::Mat illumflip;
				illum.convertTo(illumtemp, CV_32F, 1.0 / 255);
				illumtemp.copyTo(illumflip);
				flip(illumtemp, illumflip, 1);

				cv::Mat illumMask;
				diffMask.copyTo(illumMask);

				cv::threshold(illumMask, illumMask, 0.3, 1.0, cv::THRESH_TOZERO);
				//避免平均人的眉毛透露出来
				cv::Mat eyebown = cv::imread("../data/meimao.jpg", 0);
				eyebown.convertTo(eyebown, CV_32F, 1.0 / 255);
				eyebown = 1.0 - eyebown;
				illumMask = illumMask.mul(eyebown);


				illumMask = Dilation(illumMask, 13, 0);  //这个膨胀参数非常重要
				GaussianBlur(illumMask, illumMask, cv::Size(11, 11), 0);
				imshow("thresh", illumMask);

				illumMask = illumMask * 2;
				cv::Mat normFaceIllum;
				p_illum.convertTo(normFaceIllum, CV_32F, 1.0/255);

				//illum = illumtemp.mul(1.0 - illumMask) + illumflip.mul(illumMask); //对称填充，在严重不均时候有问题
				illum = illumtemp.mul(1.0 - illumMask) + normFaceIllum.mul(illumMask); //平均人填充
				illum.convertTo(illum, I_vector[2].type(), 255);
			}
			
			illumDiff.convertTo(diffMask, CV_32F, 1.0 / maxValue);
			
			I_vector[0] = illum;

			I_vector[1].convertTo(I_vector[1], CV_32F);
			I_vector[2].convertTo(I_vector[2], CV_32F);
			p_vector[1].convertTo(p_vector[1], CV_32F);
			p_vector[2].convertTo(p_vector[2], CV_32F);
			I_vector[1] = I_vector[1].mul(1.0 - diffMask) + p_vector[1].mul(diffMask);
			I_vector[2] = I_vector[2].mul(1.0 - diffMask) + p_vector[2].mul(diffMask);

			I_vector[1].convertTo(I_vector[1], CV_8U);
			I_vector[2].convertTo(I_vector[2], CV_8U);
		}

		cv::Mat apply(cv::Mat src, cv::Mat p, cv::InputArray maskMat = cv::noArray(), bool makeup = false) {
			src_ = src;
			p_ = p;
			init();
			cv::Mat I_lab, p_lab;
			cv::Mat res;
			//p_ = customCV::colorTransform(src_, p_, params_.transMask_);
			p_ = customCV::colorTransform(src_, p_);
			CV_Assert(src_.channels() == 3);
			cv::cvtColor(src_, I_lab, cv::COLOR_BGR2Lab);
			cv::cvtColor(p_, p_lab, cv::COLOR_BGR2Lab);
			std::vector<cv::Mat> I_vector, p_vector;

			cv::split(I_lab, I_vector);
			cv::split(p_lab, p_vector);
			cv::Mat I_illum, p_illum;
			I_vector[0].convertTo(I_illum, CV_64F);
			p_vector[0].convertTo(p_illum, CV_64F);
			
			if (!makeup) {
				doIllumTransform(I_illum, p_illum, I_vector, p_vector);
			}
			else {
				doMakeup(I_illum, p_illum, I_vector, p_vector);
			}
			
			cv::Mat lab;
			cv::merge(I_vector, lab);

			cvtColor(lab, res, cv::COLOR_Lab2BGR);

			return res;
		}

	public:
		Params params_;
		cv::Mat src_;
		cv::Mat p_;
		cv::Ptr<customCV::Epdfilter> epdMethod_;
		cv::Mat salancyDelta_;
	};


	cv::Ptr<IllumTransform> IllumTransform::create(int type, const Params& params) {
		if (type == MEDIAN_SUBTRACT) {
			return cv::Ptr<IllumTransform>(new IllumTransformMedianImpl(params));
		}
		else if (type == GUIDE_GUIDE_FILTER) {
			IllumTransform::algorithmType = type;
			return cv::Ptr<IllumTransform>(new IllumTransformGuideImpl(params));
		}
	}

}
