#include "photoAlgo.h"

namespace customCV {
	float skinMOG[16][7] = {
		73.53, 29.94, 17.76, 765.40, 121.44, 112.80, 0.0294,
		249.71, 233.94, 217.49, 39.94, 154.44, 396.05, 0.0331,
		161.68, 116.25, 96.95, 291.03, 60.48, 162.85, 0.0654,
		186.07, 136.62, 114.40, 274.95, 64.60, 198.27, 0.0756,
		189.26, 98.37, 51.18, 633.18, 222.40, 250.69, 0.0554,
		247.00, 152.20, 90.84, 65.23, 691.53, 609.92, 0.0314,
		150.10, 72.66, 37.76, 408.63, 200.77, 257.57, 0.0454,
		206.85, 171.09, 156.34, 530.08, 155.08, 572.79, 0.0469,
		212.78, 152.82, 120.04, 160.57, 84.52, 243.90, 0.0956,
		234.87, 175.43, 138.94, 163.80, 121.57, 279.22, 0.0763,
		151.19, 97.74, 74.59, 425.40, 73.56, 175.11, 0.1100,
		120.52, 77.55, 59.82, 330.45, 70.34, 151.82, 0.0676,
		192.20, 119.62, 82.32, 152.76, 92.14, 259.15, 0.0755,
		214.29, 136.08, 87.24, 204.90, 140.17, 270.19, 0.0500,
		99.57, 54.33, 38.06, 448.13, 90.18, 151.29, 0.0667,
		238.88, 203.08, 176.91, 178.38, 156.27, 404.99, 0.0749
	};

	float noskinMOG[16][7] = {
		254.37, 254.41, 253.82, 2.77, 2.81, 5.46, 0.0637,
		9.39, 8.09, 8.52, 46.84, 33.59, 32.48, 0.0516,
		96.57, 96.95, 91.53, 280.69, 156.79, 436.58, 0.0864,
		160.44, 162.49, 159.06, 355.98, 115.89, 591.24, 0.0636,
		74.98, 63.23, 46.33, 414.84, 245.95, 361.27, 0.0747,
		121.83, 60.88, 18.31, 2502.24, 1383.53, 237.18, 0.0365,
		202.18, 154.88, 91.04, 957.42, 1766.94, 1582.52, 0.0349,
		193.06, 201.93, 206.55, 562.88, 190.23, 447.28, 0.0649,
		51.88, 57.14, 61.55, 344.11, 191.77, 433.40, 0.0656,
		30.88, 26.84, 25.32, 222.07, 118.65, 182.41, 0.1189,
		44.97, 85.96, 131.95, 651.32, 840.52, 963.67, 0.0362,
		236.02, 236.27, 230.70, 225.03, 117.29, 331.95, 0.0849,
		207.86, 191.20, 164.12, 494.04, 237.69, 533.52, 0.0368,
		99.83, 148.11, 188.17, 955.88, 654.95, 916.70, 0.0389,
		135.06, 131.92, 123.10, 350.35, 130.30, 388.43, 0.0943,
		135.96, 103.89, 66.88, 806.44, 642.20, 350.36, 0.0477
	};

	int SkinDetector::algorithmType = 0;

	class SkinDetMogImpl : public SkinDetector {
	public:
		SkinDetMogImpl(const Params& params) {
			setParams(params);
		}
		virtual ~SkinDetMogImpl() {}
		void setParams(const Params& params) {
			
		}

		Params getParams() const {
			return params_;
		}

		cv::Mat apply(cv::Mat src) {
			cv::Mat src3f;
			src.convertTo(src3f, CV_32FC3);

			std::vector<cv::Mat> MatVec;
			cv::split(src3f, MatVec);
			cv::Mat b, g, r;
			b = MatVec[0], g = MatVec[1], r = MatVec[2];

			cv::Mat skinScore, noskinScore;
			skinScore = cv::Mat(src.size(), CV_32F, cv::Scalar(0));
			skinScore.copyTo(noskinScore);

			for (int i = 0; i < 16; i++) {
				float *p = skinMOG[i];
				cv::Mat tmp = cv::Mat(src.size(), CV_32F, cv::Scalar(0));
				tmp = (r - p[0]).mul(r - p[0]) / p[3] + (g - p[1]).mul(g - p[1]) / p[4] + (b - p[2]).mul(b - p[2]) / p[5];
				cv::Mat expmat, powmat;
				cv::exp(-tmp / 2, expmat);
				float powvalue = pow(2 * CV_PI, 1.5);
				tmp = p[6] * (expmat) / (powvalue * sqrt(p[3] * p[4] * p[5]));
				skinScore = skinScore + tmp;
			}

			for (int i = 0; i < 16; i++) {
				float *p = noskinMOG[i];
				cv::Mat tmp = cv::Mat(src.size(), CV_32F, cv::Scalar(0));
				tmp = (r - p[0]).mul(r - p[0]) / p[3] + (g - p[1]).mul(g - p[1]) / p[4] + (b - p[2]).mul(b - p[2]) / p[5];
				cv::Mat expmat, powmat;
				cv::exp(-tmp / 2, expmat);
				float powvalue = pow(2 * CV_PI, 1.5);
				tmp = p[6] * (expmat) / (powvalue * sqrt(p[3] * p[4] * p[5]));
				noskinScore = noskinScore + tmp;
			}

			cv::Mat dst = cv::Mat(src.size(), CV_8U, cv::Scalar(0));

			for (int i = 0; i < src.cols; i++) {
				for (int j = 0; j < src.rows; j++) {
					if (skinScore.at<float>(j, i) > noskinScore.at<float>(j, i)) {
						dst.at<uchar>(j, i) = 255;
					}
				}
			}
			return dst;
		}

	public:
		Params params_;
	};


	cv::Ptr<SkinDetector> SkinDetector::create(int type, const Params&  params) {
		SkinDetector::algorithmType = type;
		if (type == SKINDET_MOG) {
			return cv::Ptr<SkinDetector>(new SkinDetMogImpl(params));
		}
	}
}




