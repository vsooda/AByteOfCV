namespace customCV {
	struct AffineTransform {
		cv::Mat_<float> rotation;
		cv::Mat_<float> b;
		float c;
		AffineTransform(cv::Mat_<float> rotation_, cv::Mat_<float> b_, float c_) {
			rotation_.copyTo(rotation);
			b_.copyTo(b);
			c = c_;
		}
		cv::Mat getRotation() {
			return rotation * c;
		}

		cv::Mat getRotation_unscale() {
			return rotation;
		}
		cv::Mat getB() {
			return b;
		}

		cv::Mat operator()(const cv::Mat& locateMat) {
			cv::Mat ret;
			ret = c * rotation * locateMat;
			for (int i = 0; i < locateMat.cols; i++) {
				ret.at<float>(0, i) = ret.at<float>(0, i) + b.at<float>(0, 0);
			}
			for (int i = 0; i < locateMat.cols; i++) {
				ret.at<float>(1, i) = ret.at<float>(1, i) + b.at<float>(1, 0);
			}
			return ret;
		}
	};


	AffineTransform SimilarityTransform(const cv::Mat_<float>& sp1, const cv::Mat_<float>& sp2){
		int ptNum = sp1.rows / 2;
		double dptNum = ptNum * 1.0;
		cv::Mat_<double> shape1(2, ptNum);
		cv::Mat_<double> shape2(2, ptNum);
		for (int i = 0; i < ptNum; i++) {
			shape1(0, i) = sp1(2 * i, 0);
			shape1(1, i) = sp1(2 * i + 1, 0);
			shape2(0, i) = sp2(2 * i, 0);
			shape2(1, i) = sp2(2 * i + 1, 0);
		}
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
};
