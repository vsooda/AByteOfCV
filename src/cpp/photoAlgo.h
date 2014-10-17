#pragma once 
#include "photoUtil.h"


namespace customCV {

class PhotoBase : public cv::Algorithm
{
	//读取mask，src等等
	virtual void read() {};
	virtual void write() {};
};

class IllumTransform : public PhotoBase {
public:
	enum {
		MEDIAN_SUBTRACT = 0,
		GUIDE_GUIDE_FILTER = 1
	};
	class Params {
	public:
		Params() {
			radius_ = 31;
			symfactor_ = 0.6;
			absfactor_ = 0.2;
			laplacefactor_ = 0.2;
			eps_ = 0.01;
			p_ = cv::noArray().getMat();
			delta_ = cv::noArray().getMat();
			mask_ = cv::noArray().getMat();
			transMask_ = cv::noArray().getMat();

		}
		explicit Params(int radius, float eps,
			cv::InputArray delta = cv::noArray(), cv::InputArray mask = cv::noArray(),
			cv::InputArray transMask = cv::noArray());
		explicit Params(int radius, float symfactor, float absfactor, float laplacefactor);
		explicit Params(float radius1, float radius2);
		float symfactor_, absfactor_, laplacefactor_;

		int radius_;
		float eps_;
		cv::Mat epsMat_;
		cv::Mat delta_; //对于guidefilter来说，有delta就使用epsMat
		cv::Mat mask_;
		cv::Mat p_;
		cv::Mat transMask_;
	};
	static int algorithmType;
	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src, cv::Mat avg, cv::InputArray maskMat = cv::noArray(), bool makeup = false) = 0;

	static cv::Ptr<IllumTransform> create(int type = 0, const Params& params = Params());  //工厂函数，
};


class  Quilting : public PhotoBase {
public:
	enum {
		QUILT_TST = 0, //Image Quilting for Texture Synthesis and Transfer
		QUILT_GRAPHCUT = 1
	};
	class Params {
	public:
		Params() {
			win_ = 20;
			iter_ = 2;
			bSample_ = true;
			reduction_ = 2.0;
		}
		explicit Params(int win, int iter);  //tst参数
		int win_, iter_;

		explicit Params(bool bSample, float reduction); //graphcut 参数
		bool bSample_;
		float reduction_;
	};
	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual void apply(cv::Mat texture, cv::Mat& dst, cv::InputArray maskMat = cv::noArray()) = 0;

	static cv::Ptr<Quilting> create(int type = 0, const Params& params = Params());
};


class Salancy : public PhotoBase {
public:
	enum {
		SALANCY_CMM = 0, 
		SALANCY_CMM_ILLUM = 1   //先进行均衡化，再显著性检测
	};
	class Params {
	public:
		explicit Params(float ratio = 0.95);
		float ratio_;
		static const int DefaultNums[3];
	};

	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src, cv::InputArray maskMat = cv::noArray(), 
							cv::InputArray avgMat = cv::noArray()) = 0;
	static int algorithmType;
	static cv::Ptr<Salancy> create(int type = 0, const Params& params = Params());
};


class Inpaint : public PhotoBase {
public:
	enum {
		INPAINT_FMM = 0
	};
	class Params {
	public:
		explicit Params(int range = 5);
		int range_;
	};

	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src, cv::InputArray maskMat = cv::noArray()) = 0;

	static cv::Ptr<Inpaint> create(int type = 0, const Params& params = Params());
};

class Epdfilter : public PhotoBase {       //edge-preserving filter: guidedfilter wlsfilter binaralfilter 
public:
	enum {
		GUIDE_FILTER = 0,
		WLS_FILTER = 1
	};
	class Params {
	public:
		explicit Params(float eps = 0.01, int radius = 16);
		explicit Params(cv::Mat epsMat, int radius = 16);
		int radius_;
		float eps_;
		cv::Mat epsMat_;
	};

	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src, cv::InputArray p = cv::noArray()) = 0;

	static cv::Ptr<Epdfilter> create(int type = 0, const Params& params = Params());
};

class Makeup : public PhotoBase {
public:
	enum {
		MAKEUP_ALPHA_BLENDING = 0,
		MAKEUP_ILLUM_TRANSFORM = 1
	};
	class Params {
	public:
		Params() {
			illumParams_ = IllumTransform::Params();
		}
		explicit Params(IllumTransform::Params params);
		bool makeup_;
		IllumTransform::Params illumParams_;
	};
	static int algorithmType;
	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src, cv::Mat avg, cv::InputArray maskMat = cv::noArray()) = 0;
	static cv::Ptr<Makeup> create(int type = 0, const Params& params = Params());
};

class SkinDetector : public PhotoBase {
public:
	enum {
		SKINDET_MOG = 0
	};
	class Params {
	public:
		Params() {
		}
	};
	static int algorithmType;
	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src) = 0;
	static cv::Ptr<SkinDetector> create(int type = 0, const Params& params = Params());
};

class Tonemapping : public PhotoBase {
public:
	enum {
		TONGMAP_GUIDE = 0
	};
	class Params {
	public:
		Params() {
			epdParams_ = Epdfilter::Params();
		}
		Params(const Epdfilter::Params& params);
		customCV::Epdfilter::Params epdParams_;
	};
	static int algorithmType;
	virtual void setParams(const Params& params) = 0;
	virtual Params getParams() const = 0;
	virtual cv::Mat apply(cv::Mat src) = 0;
	static cv::Ptr<Tonemapping> create(int type = 0, const Params& params = Params());
};


}

