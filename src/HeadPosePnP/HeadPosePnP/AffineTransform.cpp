#include "AffineTransform.h"

cv::Mat pts2Mat(const std::vector<cv::Point2f> pts) {
	cv::Mat ptmat(2, pts.size(), CV_32FC1);
	for (int i = 0; i < pts.size(); i++) {
		ptmat.at<float>(0, i) = pts[i].x;
		ptmat.at<float>(1, i) = pts[i].y;
	}
	return ptmat;
}

std::vector<cv::Point2f> mat2Pts(const cv::Mat& ptmat) {
	std::vector<cv::Point2f> pts;
	for (int i = 0; i < ptmat.cols; i++) {
		float x = ptmat.at<float>(0, i);
		float y = ptmat.at<float>(1, i);
		pts.push_back(cv::Point2f(x, y));
	}
	return pts;
}


//calc the affine transform matrix with 2 x n point mat
AffineTransform SimilarityTransform(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2){
	CV_Assert(shape1.rows == 2);
	int ptNum = shape1.cols;
	double dptNum = ptNum * 1.0;
	cv::Mat_<double> rotation;
	float scale;
	rotation = cv::Mat::zeros(2, 2, CV_64FC1);
	scale = 0;

	cv::Mat_<double> mean_from = cv::Mat(2, 1, CV_64FC1, cv::Scalar(0));
	cv::Mat_<double> mean_to = cv::Mat(2, 1, CV_64FC1, cv::Scalar(0));

	cv::Mat_<double> cov = cv::Mat::zeros(2, 2, CV_64FC1);
	for (int i = 0; i < ptNum; i++) {
		mean_from(0, 0) += shape1(0, i);
		mean_from(1, 0) += shape1(1, i);
		mean_to(0, 0) += shape2(0, i);
		mean_to(1, 0) += shape2(1, i);
	}

	mean_from = mean_from / dptNum;
	mean_to = mean_to / dptNum;
	double sigma_from = 0, sigma_to = 0;
	for (int i = 0; i < ptNum; i++) {
		double diff1 = (shape1(0, i) - mean_from(0, 0)) * (shape1(0, i) - mean_from(0, 0)) + (shape1(1, i) - mean_from(1, 0)) * (shape1(1, i) - mean_from(1, 0));
		double diff2 = (shape2(0, i) - mean_to(0, 0)) * (shape2(0, i) - mean_to(0, 0)) + (shape2(1, i) - mean_to(1, 0)) * (shape2(1, i) - mean_to(1, 0));
		sigma_from += diff1;
		sigma_to += diff2;
		cv::Mat_<double> temp(2, 2);
		cv::Mat_<double> d1(2, 1), d2(2, 1);
		d1(0, 0) = shape1(0, i) - mean_from(0, 0);
		d1(1, 0) = shape1(1, i) - mean_from(1, 0);
		d2(0, 0) = shape2(0, i) - mean_to(0, 0);
		d2(1, 0) = shape2(1, i) - mean_to(1, 0);
		temp = d1 * d2.t();
		cov = cov + temp;
	}

	sigma_from = sigma_from / dptNum;
	sigma_to = sigma_to / dptNum;
	cov = cov / dptNum;
	////std::cout << sigma_from << std::endl;
	////std::cout << sigma_to << std::endl;
	cv::SVD svd;
	cv::Mat u, vt, d, s, w;
	//matlab diff form opencv http://stackoverflow.com/questions/12029486/matlab-svd-output-in-opencv
	//svd.compute(cov, d, u, vt, cv::SVD::FULL_UV);
	cv::SVDecomp(cov, w, u, vt, cv::SVD::FULL_UV);
	d = cv::Mat(cov.size(), CV_64FC1, cv::Scalar(0));
	for (int i = 0; i < w.rows; i++) {
		d.at<double>(i, i) = w.at<double>(i, 0);
	}
	////std::cout << std::endl;
	////std::cout << cov << std::endl;
	////std::cout << u << std::endl;
	////std::cout << d << std::endl;
	////std::cout << vt << std::endl;
	s = cv::Mat::eye(cov.size(), CV_64FC1);

	if (cv::determinant(cov) < 0) {
		if (d.at<double>(1, 1) < d.at<double>(0, 0))
			s.at<double>(1, 1) = -1;
		else
			s.at<double>(0, 0) = -1;
	}

	cv::Mat r(2, 2, CV_64FC1);
	r = u * s * vt;
	double c = 1;
	if (sigma_from != 0) {
		c = 1.0 / sigma_from * cv::trace(d * s)[0];
	}

	//r = c * r;
	r = r.t();
	cv::Mat t = mean_to - c * r * mean_from;

	//std::cout << r << std::endl << std::endl;;

	t.convertTo(t, CV_32F);
	r.convertTo(r, CV_32F);
	return AffineTransform(r, t, c);
}


AffineTransform SimilarityTransformPts(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) {
	cv::Mat shape1, shape2;
	shape1 = pts2Mat(pts1);
	shape2 = pts2Mat(pts2);
	return SimilarityTransform(shape1, shape2);
}

std::vector<cv::Point2f> estimate2dRotate(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, float *pangle) {
	AffineTransform atf = SimilarityTransformPts(pts1, pts2);
	if (pangle != NULL) {
		float angle = asin(atf.getRotation_unscale().at<float>(0, 1)) * 180.0 / CV_PI;
		*pangle = angle;
	}
	cv::Mat tform = atf.getRotation_unscale();
	cv::Mat fromPtmat = pts2Mat(pts1);
	cv::Mat rotateMat = tform * fromPtmat;
	cv::Mat b = atf.getB();
	//cv::Mat b = atf.getB_unscale();
	/*for (int col = 0; col < rotateMat.cols; col++) {
	rotateMat.col(col) = rotateMat.col(col) + b;
	}*/
	return mat2Pts(rotateMat);
}

std::vector<cv::Point2f> estimate2dRotate(std::vector<cv::Point2f>& queryPts, const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, float *pangle) {
	AffineTransform atf = SimilarityTransformPts(pts1, pts2);
	if (pangle != NULL) {
		float angle = asin(atf.getRotation_unscale().at<float>(0, 1)) * 180.0 / CV_PI;
		*pangle = angle;
	}
	cv::Mat tform = atf.getRotation_unscale();
	cv::Mat fromPtmat = pts2Mat(queryPts);
	cv::Mat rotateMat = tform * fromPtmat;
	cv::Mat b = atf.getB();
	return mat2Pts(rotateMat);
}

float estimate2dRotateAngle(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2) {
	AffineTransform atf = SimilarityTransformPts(pts1, pts2);
	float angle = asin(atf.getRotation_unscale().at<float>(0, 1)) * 180.0 / CV_PI;
	return angle;
}

std::vector<cv::Point2f> estimate2dRotateDelta(const std::vector<cv::Point2f>& pts1, const std::vector<cv::Point2f>& pts2, std::vector<cv::Point2f> deltaPts) {
	AffineTransform atf = SimilarityTransformPts(pts1, pts2);
	cv::Mat tform = atf.getRotation();
	cv::Mat deltaPtmat = pts2Mat(deltaPts);
	cv::Mat deltaMat = tform * deltaPtmat;
	std::vector<cv::Point2f> pts = mat2Pts(deltaMat);
	for (int i = 0; i < deltaPts.size(); i++) {
		pts[i].x = pts[i].x + deltaPts[i].x;
		pts[i].y = pts[i].y + deltaPts[i].y;
	}
	return pts;
}