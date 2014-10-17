#include "photoAlgo.h"

namespace customCV {
	int Makeup::algorithmType = 0;

	Makeup::Params::Params(IllumTransform::Params params) {
		illumParams_ = params;
	}

	class MakeupGGIlumImpl : public Makeup {
	public:
		MakeupGGIlumImpl(const Params& params) {
			setParams(params);
		}
		virtual ~MakeupGGIlumImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}

		Params getParams() const {
			return params_;
		}

		virtual cv::Mat apply(cv::Mat src, cv::Mat avg, cv::InputArray maskMat = cv::noArray()) {
			if (Makeup::algorithmType == MAKEUP_ILLUM_TRANSFORM) {
				setIllumAlgorithm(IllumTransform::create(IllumTransform::GUIDE_GUIDE_FILTER, params_.illumParams_));
				return illumMethod_->apply(src, avg, maskMat, true);
			}
		}

		void setIllumAlgorithm(cv::Ptr<IllumTransform> itf) {
			illumMethod_ = itf;
		}

	public:
		Params params_;
		cv::Ptr<customCV::IllumTransform> illumMethod_;
	};


	cv::Ptr<Makeup> Makeup::create(int type, const Params&  params) {
		Makeup::algorithmType = type;
		if (type == MAKEUP_ILLUM_TRANSFORM) {
			return cv::Ptr<Makeup>(new MakeupGGIlumImpl(params));
		}
	}



}