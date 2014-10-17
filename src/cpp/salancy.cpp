#include "photoAlgo.h"
#include "photoUtil.h"
using namespace std;
using namespace cv;

namespace customCV {

	int Salancy::algorithmType = 0;

	template<typename T> inline T sqr(T x) { return x * x; } // out of range risk for T = byte, ...

	template<class T, int D>
	inline T vecSqrDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2) 
	{
		T s = 0; for (int i = 0; i < D; i++) s += sqr(v1[i] - v2[i]); return s; 
	} // out of range risk for T = byte, ...

	template<class T, int D>
	inline T vecDist(const cv::Vec<T, D> &v1, const cv::Vec<T, D> &v2)
	{
		return sqrt(vecSqrDist(v1, v2)); 
	} // out of range risk for T = byte, ...


	Salancy::Params::Params(float ratio)  {
		ratio_ = ratio;
	}
	int const Salancy::Params::DefaultNums[3] = { 12, 12, 12 };

	class SalancyCmmImpl : public Salancy {
	public:
		SalancyCmmImpl(const Params& params) {
			setParams(params);
		}
		virtual ~SalancyCmmImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		cv::Mat apply(cv::Mat src, cv::InputArray maskMat = cv::noArray(),
						cv::InputArray avgMat = cv::noArray()) {
			cv::Mat img;
			src.copyTo(img);
			cv::Mat mask = maskMat.getMat();
			if (mask.empty()) {
				mask = cv::Mat(src.size(), CV_8U, cv::Scalar(255)); //没有mask，就加入mask
			}
			cv::Mat avg = avgMat.getMat();
			if (algorithmType == SALANCY_CMM_ILLUM) {
				CV_Assert(avg.data != NULL);
				img.convertTo(img, CV_8UC3, 255.0);
				customCV::IllumTransform::Params params(31, 0.6, 0.2, 0.2);
				cv::Ptr<customCV::IllumTransform> illumMethod =
						customCV::IllumTransform::create(customCV::IllumTransform::MEDIAN_SUBTRACT, params);
				img = illumMethod->apply(img, avg, mask);
				imshow("img", img);
				img.convertTo(img, src.type(), 1.0/255.0);
			}
			cv::Mat img3f = img;
			// Quantize colors and
			Mat idx1i, binColor3f, colorNums1i, _colorSal;
			D_Quantize(img3f, idx1i, binColor3f, colorNums1i, 0.95, Salancy::Params::DefaultNums, mask);
			cvtColor(binColor3f, binColor3f, CV_BGR2Lab);

			GetHC(binColor3f, colorNums1i, _colorSal);
			float* colorSal = (float*)(_colorSal.data);
			Mat salHC1f(img3f.size(), CV_32F, cv::Scalar(0));
			for (int r = 0; r < img3f.rows; r++){
				float* salV = salHC1f.ptr<float>(r);
				int* _idx = idx1i.ptr<int>(r);
				for (int c = 0; c < img3f.cols; c++) {
					salV[c] = colorSal[_idx[c]];
				}
			}

			GaussianBlur(salHC1f, salHC1f, Size(3, 3), 0);
			//normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX, -1, mask);
			normalize(salHC1f, salHC1f, 0, 1, NORM_MINMAX);
			for (int i = 0; i < salHC1f.cols; i++) {
				for (int j = 0; j < salHC1f.rows; j++) {
					if (!mask.at<uchar>(j, i)) {
						salHC1f.at<float>(j, i) = 0;
					}
				}
			}
			return salHC1f;
		}

		int D_Quantize(cv::Mat& img3f, cv::Mat &idx1i, cv::Mat &_color3f, cv::Mat &_colorNum, double ratio = 0.95, const int clrNums[3] = Salancy::Params::DefaultNums, cv::InputArray maskMat = cv::noArray())
		{
			cv::Mat mask = maskMat.getMat();
			if (mask.empty()) {
				mask = cv::Mat(img3f.size(), CV_8U, cv::Scalar(255)); //没有mask，就加入mask
			}
			float clrTmp[3] = { clrNums[0] - 0.0001f, clrNums[1] - 0.0001f, clrNums[2] - 0.0001f };
			int w[3] = { clrNums[1] * clrNums[2], clrNums[2], 1 };

			CV_Assert(img3f.data != NULL);
			idx1i = Mat::zeros(img3f.size(), CV_32S);
			int rows = img3f.rows, cols = img3f.cols;
			if (img3f.isContinuous() && idx1i.isContinuous() && mask.isContinuous()){
				cols *= rows;
				rows = 1;
			}
			int cnt = 0;
			// Build color pallet 调色板
			for (int y = 0; y < rows; y++)	{
				const float* imgData = img3f.ptr<float>(y);
				int* idx = idx1i.ptr<int>(y);
				uchar* maskPtr = mask.ptr<uchar>(y);
#pragma omp parallel for
				for (int x = 0; x < cols; x++, imgData += 3) {
					if (maskPtr[x] == 0) {
						cnt++; continue;
					}
					idx[x] = (int)(imgData[0] * clrTmp[0])*w[0] + (int)(imgData[1] * clrTmp[1])*w[1] + (int)(imgData[2] * clrTmp[2]);
				}
			}
			map<int, int> pallet;
			for (int y = 0; y < rows; y++)	{
				int* idx = idx1i.ptr<int>(y);
				uchar* maskPtr = mask.ptr<uchar>(y);
				for (int x = 0; x < cols; x++) {
					if (maskPtr[x] == 0) {
						continue;
					}
					pallet[idx[x]] ++;
				}
			}

			// Find significant colors
			int maxNum = 0; {
				int count = 0;
				vector<pair<int, int> > num; // (num, color) pairs in num
				num.reserve(pallet.size());
				for (map<int, int>::iterator it = pallet.begin(); it != pallet.end(); it++)
					num.push_back(pair<int, int>(it->second, it->first)); // (color, num) pairs in pallet
				sort(num.begin(), num.end(), std::greater<pair<int, int> >());

				maxNum = (int)num.size();
				int maxDropNum = cvRound(rows * cols * (1 - ratio));
				for (int crnt = num[maxNum - 1].first; crnt < maxDropNum && maxNum > 1; maxNum--)
					crnt += num[maxNum - 2].first;
				maxNum = min(maxNum, 256); // To avoid very rarely case
				if (maxNum <= 10)
					maxNum = min(10, (int)num.size());

				pallet.clear();
				for (int i = 0; i < maxNum; i++)
					pallet[num[i].second] = i;

				vector<Vec3i> color3i(num.size());
				for (unsigned int i = 0; i < num.size(); i++) {
					color3i[i][0] = num[i].second / w[0];
					color3i[i][1] = num[i].second % w[0] / w[1];
					color3i[i][2] = num[i].second % w[1];
				}

				for (unsigned int i = maxNum; i < num.size(); i++)	{
					int simIdx = 0, simVal = INT_MAX;
					for (int j = 0; j < maxNum; j++) {
						int d_ij = vecSqrDist(color3i[i], color3i[j]);
						if (d_ij < simVal)
							simVal = d_ij, simIdx = j;
					}
					pallet[num[i].second] = pallet[num[simIdx].second];
				}
			}

			_color3f = Mat::zeros(1, maxNum, CV_32FC3);
			_colorNum = Mat::zeros(_color3f.size(), CV_32S);

			Vec3f* color = (Vec3f*)(_color3f.data);
			int* colorNum = (int*)(_colorNum.data);
			for (int y = 0; y < rows; y++) {
				const Vec3f* imgData = img3f.ptr<Vec3f>(y);
				int* idx = idx1i.ptr<int>(y);
				uchar* maskPtr = mask.ptr<uchar>(y);
#pragma omp parallel for
				for (int x = 0; x < cols; x++)	{
					if (maskPtr[x] == 0) {
						continue;
					}
					idx[x] = pallet[idx[x]];
					color[idx[x]] += imgData[x];
					colorNum[idx[x]] ++;
				}
			}
			for (int i = 0; i < _color3f.cols; i++)
				color[i] /= colorNum[i];

			return _color3f.cols;
		}

		void SmoothSaliency(Mat &colorNum1i, Mat &sal1f, float delta, const vector<vector<pair<float, int> > > similar)
        {
			if (sal1f.cols < 2)
				return;
			CV_Assert(sal1f.rows == 1 && sal1f.type() == CV_32FC1);
			CV_Assert(colorNum1i.size() == sal1f.size() && colorNum1i.type() == CV_32SC1);

			int binN = sal1f.cols;
			Mat newSal1d = Mat::zeros(1, binN, CV_64FC1);
			float *sal = (float*)(sal1f.data);
			double *newSal = (double*)(newSal1d.data);
			int *pW = (int*)(colorNum1i.data);

			// Distance based smooth
			int n = max(cvRound(binN * delta), 2);
			vector<double> dist(n, 0), val(n), w(n);
			for (int i = 0; i < binN; i++){
				const vector<pair<float, int> > &similari = similar[i];
				double totalDist = 0, totoalWeight = 0;
				for (int j = 0; j < n; j++){
					int ithIdx = similari[j].second;
					dist[j] = similari[j].first;
					val[j] = sal[ithIdx];
					w[j] = pW[ithIdx];
					totalDist += dist[j];
					totoalWeight += w[j];
				}
				double valCrnt = 0;
				for (int j = 0; j < n; j++)
					valCrnt += val[j] * (totalDist - dist[j]) * w[j];

				newSal[i] = valCrnt / (totalDist * totoalWeight);
			}
			normalize(newSal1d, sal1f, 0, 1, NORM_MINMAX, CV_32FC1);
		}

		void SmoothSaliency(Mat &sal1f, float delta, const vector<vector<pair<float, int> > > &similar)
		{
			Mat colorNum1i = Mat::ones(sal1f.size(), CV_32SC1);
			SmoothSaliency(colorNum1i, sal1f, delta, similar);
		}

		void GetHC(Mat &binColor3f, Mat &colorNums1i, Mat &_colorSal)
		{
			Mat weight1f;
			normalize(colorNums1i, weight1f, 1, 0, NORM_L1, CV_32F);

			int binN = binColor3f.cols;
			_colorSal = Mat::zeros(1, binN, CV_32F);
			float* colorSal = (float*)(_colorSal.data);
			vector<vector<pair<float, int> > > similar(binN); // Similar color: how similar and their index
			Vec3f* color = (Vec3f*)(binColor3f.data);
			float *w = (float*)(weight1f.data);
			for (int i = 0; i < binN; i++){
				vector<pair<float, int> > &similari = similar[i];
				similari.push_back(make_pair(0.f, i));
				for (int j = 0; j < binN; j++){
					if (i == j)
						continue;
					float dij = vecDist<float, 3>(color[i], color[j]);
					similari.push_back(make_pair(dij, j));
					colorSal[i] += w[j] * dij;
				}
				sort(similari.begin(), similari.end());
			}

			SmoothSaliency(_colorSal, 0.25f, similar);
		}

		void D_Recover(cv::Mat& idx1i, cv::Mat &img3f, cv::Mat &color3f)
		{
			CV_Assert(idx1i.data != NULL);
			img3f.create(idx1i.size(), CV_32FC3);

			Vec3f* color = (Vec3f*)(color3f.data);
			for (int y = 0; y < idx1i.rows; y++) {
				Vec3f* imgData = img3f.ptr<Vec3f>(y);
				const int* idx = idx1i.ptr<int>(y);
				for (int x = 0; x < idx1i.cols; x++) {
					imgData[x] = color[idx[x]];
					CV_Assert(idx[x] < color3f.cols);
				}
			}
		}

	public:
		Params params_;
	};

	cv::Ptr<Salancy> Salancy::create(int type, const Params& params) {
		if (type == SALANCY_CMM) {
			return cv::Ptr<Salancy>(new SalancyCmmImpl(params));
		}
		else if (type == SALANCY_CMM_ILLUM) {
			algorithmType = SALANCY_CMM_ILLUM;
			return cv::Ptr<Salancy>(new SalancyCmmImpl(params));
		}
	}
}
