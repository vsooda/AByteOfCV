#include "photoAlgo.h"
using namespace cv;
using namespace std;

class PhotoBaseTest {
public:
	cv::Mat src_, dst_, mask_, avg_, extra1_, extra2_;
	int algoType;

	PhotoBaseTest() {
		algoType = 0;
	}

	virtual void doProcess() = 0;

	virtual void setAlgorithmType(int type) {
		algoType = type;
	}

	cv::Mat apply(const char* srcName, const char* maskName = NULL, const char* avgName = NULL,
		const char* extraName1 = NULL, const char* extraName2 = NULL) {
		init(srcName, maskName, avgName, extraName1);
		//cv::Size sz = src_.size();
		//resize(src_, src_, mask_.size());
		clock_t a, b;
		a = clock();
		doProcess();
		b = clock();
		cout << "time: " << (b - a) * 1000.0 / CLOCKS_PER_SEC << " ms" << endl;
		showResult();
		//resize(dst_, dst_, sz);
		return dst_;
	}

	//extra1表示delta
	void init(const char* srcName, const char* maskName = NULL, const char* avgName = NULL,
			const char* extraName1 = NULL, const char* extraName2 = NULL) {
		src_ = imread(srcName);
		CV_Assert(src_.data != NULL);
		if (maskName) {
			mask_ = imread(maskName, 0);
			CV_Assert(mask_.data != NULL);
		}
		if (avgName) {
			avg_ = imread(avgName);
			CV_Assert(avg_.data != NULL);
		}
		if (extraName1) {
			extra1_ = imread(extraName1, 0);
			CV_Assert(extra1_.data != NULL);
		}
		if (extraName2) {
			extra2_ = imread(extraName2);
			CV_Assert(extra2_.data != NULL);
		}
	}

	virtual void showResult() {
		imshow("src", src_);
		imshow("dst", dst_);
		waitKey();
	}
};

/*
	TonemapTest tt;
	tt.apply("../data/2.jpg");
*/
class TonemapTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		customCV::Epdfilter::Params params(0.001, 20);
		cv::Ptr<customCV::Tonemapping> tonemap = customCV::Tonemapping::create(customCV::Tonemapping::TONGMAP_GUIDE, params);
		dst_ = tonemap->apply(src_);
	}
};

/*
	IllumtrasformTest it;
	it.setAlgorithmType(1);//可选
	it.apply("../data/i3.jpg", NULL, "../data/std_512.jpg");
*/
class IllumtrasformTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		if (algoType == customCV::IllumTransform::MEDIAN_SUBTRACT) {
			customCV::IllumTransform::Params params(31, 0.6, 0.2, 0.2);
			cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create(algoType, params);
			dst_ = illumMethod->apply(src_, avg_);
		}
		else if (algoType == customCV::IllumTransform::GUIDE_GUIDE_FILTER) {
			customCV::IllumTransform::Params params(30, 0.2, extra1_, mask_);
			cv::Ptr<customCV::IllumTransform> illumMethod = customCV::IllumTransform::create(algoType, params);
			dst_ = illumMethod->apply(src_, avg_);
		}
	}
};

class MakeupTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		customCV::IllumTransform::Params illumparams(30, 0.2, extra1_, mask_);
		customCV::Makeup::Params params(illumparams);
		cv::Ptr<customCV::Makeup> illumMethod = customCV::Makeup::create(customCV::Makeup::MAKEUP_ILLUM_TRANSFORM, params);
		dst_ = illumMethod->apply(src_, avg_);
	}
};

class QuiltTest : public PhotoBaseTest {
public:
	int size_, w_, iter_;
	virtual void doProcess() {
		if (algoType == customCV::Quilting::QUILT_TST) {
			dst_ = cv::Mat(size_, size_, CV_32FC3, cv::Scalar(0, 0, 0));
			src_.convertTo(src_, CV_32FC3, 1.0 / 255.0);
			customCV::Quilting::Params params(w_, iter_);
			cv::Ptr<customCV::Quilting> quiltMethod = customCV::Quilting::create(algoType, params);
			//cv::Ptr<customCV::Quilting> quiltMethod = customCV::Quilting::create(algoType);
			quiltMethod->apply(src_, dst_);
			dst_.convertTo(dst_, CV_8UC3, 255);
		}
		else if (algoType == customCV::Quilting::QUILT_GRAPHCUT) {
			dst_ = avg_;
			customCV::Quilting::Params params(true, 1.2);
			cv::Ptr<customCV::Quilting> quiltMethod = customCV::Quilting::create(algoType, params);
			//cv::Ptr<customCV::Quilting> quiltMethod = customCV::Quilting::create(algoType);
			quiltMethod->apply(src_, dst_, mask_);
		}
	}
	void setExtraParams(int size, int w, int iter) {
		size_ = size;
		w_ = w;
		iter_ = iter;
	}
};



class SkinDetectorTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		cv::Ptr<customCV::SkinDetector> skinMethod_ = customCV::SkinDetector::create(customCV::SkinDetector::SKINDET_MOG);
		dst_ = skinMethod_->apply(src_);
	}
};

class SalancyTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		src_.convertTo(src_, CV_32FC3, 1.0 / 255.0);
		cv::Ptr<customCV::Salancy> salancyMethod = customCV::Salancy::create(algoType);
		dst_ = salancyMethod->apply(src_, mask_, avg_);
		dst_.convertTo(dst_, CV_8UC1, 255);
	}
};

class EpdfilterTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		cv::Ptr<customCV::Epdfilter> epdMethod = customCV::Epdfilter::create(customCV::Epdfilter::GUIDE_FILTER);
		dst_ = epdMethod->apply(src_, src_);
	}
};

class InpaintTest : public PhotoBaseTest {
public:
	virtual void doProcess() {
		customCV::Inpaint::Params params(10);
		cv::Ptr<customCV::Inpaint> inpaintMethod = customCV::Inpaint::create(customCV::Inpaint::INPAINT_FMM, params);
		dst_ = inpaintMethod->apply(src_, mask_);
	}
};
