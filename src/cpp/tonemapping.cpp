#include "photoAlgo.h"

namespace customCV {
	int Tonemapping::algorithmType = 0;
	Tonemapping::Params::Params(const Epdfilter::Params& params) {
		epdParams_ = params;
	}

	class TonemapGuideImpl : public Tonemapping {
	public:
		TonemapGuideImpl(const Params& params) {
			setParams(params);
		}
		virtual ~TonemapGuideImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}

		Params getParams() const {
			return params_;
		}

		void setEpdAlgorithm(cv::Ptr<Epdfilter> epd) {
			epdMethod_ = epd;
		}

		cv::Mat apply(cv::Mat src) {
			double contrast = 5.0;
			cv::Mat  dst;
			cv::Mat src32f;
			cv::Mat iten;
			src.convertTo(src32f, CV_32FC3);
			cvConvertScale(&((IplImage)src32f), &((IplImage)src32f), 1.0 / 255);

			iten = cv::Mat(src.rows, src.cols, CV_32F, cv::Scalar(0));
			for (int i = 0; i < src.cols; i++) {
				for (int j = 0; j < src.rows; j++) {
					cv::Vec3f temp = src32f.at<cv::Vec3f>(j, i);
					double t = temp[0] + temp[1] * 40 + temp[2] * 20;
					double val = t / 61 + 0.001;

					iten.at<float>(j, i) = val;
				}
			}

			cv::Mat iten32f, logI, logF, F32f;

			double minF, maxF;
			//将logI转化到0，1范围内，再引导滤波

			log(iten, logI);

			setEpdAlgorithm(Epdfilter::create(Epdfilter::GUIDE_FILTER, params_.epdParams_));
			logF = epdMethod_->apply(logI, logI);
			logF.convertTo(logF, CV_32F);

			cv::minMaxLoc(logF, &minF, &maxF, 0, 0);

			cv::Mat detail;
			detail = logI - logF;

			double delta = maxF - minF;

			double gamma = log(contrast) / delta;

			cv::Mat N;
			exp((logF*gamma + detail), N);


			cv::Mat scaleMat, scaleMat3;
			cv::divide(N, iten, scaleMat);
			cv::Mat scaleMats[] = { scaleMat, scaleMat, scaleMat };
			cv::merge(scaleMats, 3, scaleMat3);

			dst = src32f.mul(scaleMat3);
			double a, b;
			cv::minMaxLoc(scaleMat, &a, &b, 0, 0);
			return dst;
		}
	public:
		Params params_;
		cv::Ptr<customCV::Epdfilter> epdMethod_;
	};

	cv::Ptr<Tonemapping> Tonemapping::create(int type, const Params& params) {
		if (type == TONGMAP_GUIDE) {
			return cv::Ptr<Tonemapping>(new TonemapGuideImpl(params));
		}
	}

}
