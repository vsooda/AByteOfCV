#include "photoAlgo.h"
#include "photoUtil.h"
using namespace std;
using namespace cv;

namespace customCV {

	Epdfilter::Params::Params(float eps, int radius) {
		radius_ = radius;
		eps_ = eps;
	}

	Epdfilter::Params::Params(cv::Mat epsMat, int radius) {
		radius_ = radius;
		epsMat_ = epsMat;
		eps_ = -1; //表示使用epsMat而不是eps
	}


	class EpdfilterGuideImpl : public Epdfilter {
	public:
		EpdfilterGuideImpl(const Params& params) {
			setParams(params);
		}
		virtual ~EpdfilterGuideImpl() {
		}
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		void init() {
			if (params_.eps_ > 0) {
				params_.epsMat_ = cv::Mat(src_.size(), CV_32F, cv::Scalar(params_.eps_));
			}
		}

		//guide filter 的实现。。应该转到0,1才有效果吧？？
		cv::Mat doGuidefilter(cv::Mat src, cv::Mat guide, int r, cv::Mat eps)
		{
			cv::Mat I;
			cv::Mat p;
			src.convertTo(I, CV_64FC1);
			guide.convertTo(p, CV_64FC1);
			eps.convertTo(eps, CV_64FC1);
			int hei = I.rows;
			int wid = I.cols;
			cv::Mat N;
			cv::boxFilter(cv::Mat::ones(hei, wid, I.type()), N, CV_64FC1, cv::Size(r, r));
			cv::Mat mean_I;
			cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(r, r));
			cv::Mat mean_p;
			cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(r, r));
			cv::Mat mean_Ip;
			cv::boxFilter(I.mul(p), mean_Ip, CV_64FC1, cv::Size(r, r));
			cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
			cv::Mat mean_II;
			cv::boxFilter(I.mul(I), mean_II, CV_64FC1, cv::Size(r, r));
			cv::Mat var_I = mean_II - mean_I.mul(mean_I);

			cv::Mat a = cov_Ip / (var_I + eps);
			cv::Mat b = mean_p - a.mul(mean_I);
			cv::Mat mean_a;
			cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r));
			mean_a = mean_a / N;
			cv::Mat mean_b;
			cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r));
			mean_b = mean_b / N;
			cv::Mat q = mean_a.mul(I) + mean_b;
			q.convertTo(q, src.type());
			return q;
		}

		cv::Mat apply(cv::Mat src, cv::InputArray p) {
			src_ = src;
			p_ = p.getMat();
			init();
			if (src.channels() == 1 && p_.channels() == 1) {
				return doGuidefilter(src_, p_, params_.radius_, params_.epsMat_);
			}
			else if (src.channels() == 3 && p_.channels() == 3) {
				vector<cv::Mat>srcVec, pVec, dstVec;
				cv::split(src_, srcVec);
				cv::split(p_, pVec);
				dstVec = srcVec;
				dstVec[0] = doGuidefilter(srcVec[0], pVec[0], params_.radius_, params_.epsMat_);
				dstVec[1] = doGuidefilter(srcVec[1], pVec[1], params_.radius_, params_.epsMat_);
				dstVec[2] = doGuidefilter(srcVec[2], pVec[2], params_.radius_, params_.epsMat_);
				cv::Mat dst;
				cv::merge(dstVec, dst);
				return dst;
			}
			else {
				CV_Error(CV_StsOutOfRange, "the image's channels in guidefilter must 1 or 3 and they are same");
			}
		}

	public:
		Params params_;
		cv::Mat src_;
		cv::Mat p_;
	};

	cv::Ptr<Epdfilter> Epdfilter::create(int type, const Params& params) {
		if (type == GUIDE_FILTER) {
			return cv::Ptr<Epdfilter>(new EpdfilterGuideImpl(params));
		}
	}

}