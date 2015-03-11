#include "ESR_shape.h"

EsrShape::EsrShape(const char* faceName, const char* shapeName, int landmarkNum)
{
	_landnum = landmarkNum;
	dlib::deserialize(faceName) >> _detector;
	dlib::deserialize(shapeName) >> _sp;
	setInitPts();
	setRotateEstimateIndexs68();
}

EsrShape::~EsrShape() {
}

void EsrShape::setRotateEstimateIndexs(std::vector<int> indexs) {
	_rotateEstimateIndexs.clear();
	for (int i = 0; i < indexs.size(); i++) {
		_rotateEstimateIndexs.push_back(indexs[i]);
	}
}

cv::Rect EsrShape::getFaceRect() {
	return _rect;
}

void EsrShape::setRotateEstimateIndexs74() {
	_rotateEstimateIndexs.clear();
	for (int i = 29; i <= 45; i++) {
		_rotateEstimateIndexs.push_back(i);
	}
	for (int i = 65; i <= 73; i++) {
		_rotateEstimateIndexs.push_back(i);
	}
}

void EsrShape::setRotateEstimateIndexs68() {
	_rotateEstimateIndexs.clear();
	for (int i = 27; i <= 47; i++) {
		_rotateEstimateIndexs.push_back(i);
	}
	_rotateEstimateIndexs.push_back(7);
	_rotateEstimateIndexs.push_back(0);
	_rotateEstimateIndexs.push_back(14);
}

std::vector<cv::Point2f> EsrShape::getIndexPts(const std::vector<cv::Point2f> &pts, std::vector<int> indexs) {
	std::vector<cv::Point2f> indexPts;
	for (int i = 0; i < indexs.size(); i++) {
		indexPts.push_back(pts[indexs[i]]);
	}
	return indexPts;
}

void EsrShape::setInitPts() {
	_initPts.clear();
	std::vector<float> initVector = _sp.getInitShapeVector(_landnum);
	CV_Assert(initVector.size() == 2 * _landnum);
	for (int i = 0; i < _landnum; i++) {
		_initPts.push_back(cv::Point2f(initVector[2 * i], initVector[2 * i + 1]));
	}
}

std::vector<cv::Point2f> EsrShape::getRotatePts(std::vector<cv::Point2f> pts, std::vector<cv::Point2f> initPts, float* pangle) {
	return estimate2dRotate(pts, initPts, pangle);
}

std::vector<cv::Point2f> EsrShape::getRotatePts(std::vector<cv::Point2f> queryPts, std::vector<cv::Point2f> pts,
	std::vector<cv::Point2f> initPts, float* pangle) {
	return estimate2dRotate(queryPts, pts, initPts, pangle);
}

cv::Mat EsrShape::getRotateMat(const cv::Mat& src, std::vector<cv::Point2f> pts, std::vector<cv::Point2f> initPts, float* pangle) {
	float angle;
	estimate2dRotate(pts, initPts, &angle);
	if (pangle != NULL) {
		*pangle = angle;
	}
	cv::Point2f rect_center(_rect.x + _rect.width / 2, _rect.y + _rect.height / 2);
	cv::Mat rot_mat = cv::getRotationMatrix2D(cv::Point2f(0, 0), angle, 1.0);
	cv::Mat dst;
	cv::warpAffine(src, dst, rot_mat, src.size());
	return dst;
}



bool EsrShape::detectAndRotate(const cv::Mat& src, cv::Mat& rotatedImage) {
	if (!detect(src)) {
		return false;
	}
	rotatedImage = getRotateMat(src, _pts, _initPts);
	_pts = getRotatePts(_pts, _initPts);
	return true;
}

bool  EsrShape::detect(const cv::Mat& src)
{
	dlib::array2d<dlib::rgb_pixel> img;
	std::vector<cv::Rect> faces;
	dlib::cv_image<dlib::rgb_pixel> *pimg = new dlib::cv_image<dlib::rgb_pixel>(src);
	assign_image(img, *pimg);
	_pts.clear();

	std::vector<dlib::rectangle> dets;
	dets = _detector(img);
	if (dets.size() < 1) {
		return false;
	}
	_rect = cv::Rect(cv::Point(dets[0].left(), dets[0].top()), cv::Point(dets[0].right(), dets[0].bottom()));
	dlib::rectangle det(_rect.x, _rect.y, _rect.x + _rect.width, _rect.y + _rect.height);
	dlib::full_object_detection shape = _sp(img, det);

	for (int i = 0; i < shape.num_parts(); i++) {
		dlib::point pt = shape.part(i);
		_pts.push_back(cv::Point2f(pt.x(), pt.y()));
	}
	delete pimg;
	return true;
}



void EsrShape::similarity_transform_correct(std::vector<cv::Point2f>& query_pts, const std::vector<cv::Point2f>& detect_pts, const std::vector<cv::Point2f>& correct_pts) {

	std::vector<dlib::vector<float, 2> > select_points, correct_points;
	for (int i = 0; i < detect_pts.size(); i++) {
		select_points.push_back(dlib::point(detect_pts[i].x, detect_pts[i].y));
		correct_points.push_back(dlib::point(correct_pts[i].x, correct_pts[i].y));
	}


	dlib::point_transform_affine tform = find_similarity_transform(select_points, correct_points);

	for (int i = 0; i < query_pts.size(); i++) {
		dlib::point p = tform(dlib::point(query_pts[i].x, query_pts[i].y));
		query_pts[i].x = p.x();
		query_pts[i].y = p.y();
	}
}


void EsrShape::detectWithRect(const cv::Mat&  src, cv::Rect rect) {
	_rect = rect;

	dlib::array2d<dlib::rgb_pixel> img;
	std::vector<cv::Rect> faces;
	dlib::cv_image<dlib::rgb_pixel> *pimg = new dlib::cv_image<dlib::rgb_pixel>(src);
	assign_image(img, *pimg);
	_pts.clear();

	dlib::rectangle det(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
	dlib::full_object_detection shape = _sp(img, det);

	for (int i = 0; i < shape.num_parts(); i++) {
		dlib::point pt = shape.part(i);
		_pts.push_back(cv::Point2f(pt.x(), pt.y()));
	}
	delete pimg;
}

void EsrShape::detectAndRotateWithRect(const cv::Mat&  src, cv::Rect rect, cv::Mat& rotatedImage) {
	detectWithRect(src, rect);
	rotatedImage = getRotateMat(src, _pts, _initPts);
	_pts = getRotatePts(_pts, _initPts);
}

void EsrShape::rotateMatAndPts(cv::Mat& src, std::vector<cv::Point2f> &pts, float* pangle) {
	std::vector<cv::Point2f> initPts = filter74to68(_initPts);
	src = getRotateMat(src, pts, initPts, pangle);
	pts = getRotatePts(pts, initPts);
}


void EsrShape::indexRotateMatAndPts(cv::Mat& src, std::vector<cv::Point2f> &pts, float* pangle) {
	std::vector<cv::Point2f> initPts = getIndexPts(filter74to68(_initPts), _rotateEstimateIndexs);
	std::vector<cv::Point2f> indexPts = getIndexPts(pts, _rotateEstimateIndexs);
	src = getRotateMat(src, indexPts, initPts, pangle);
	pts = getRotatePts(pts, indexPts, initPts);
}


void  EsrShape::draw(cv::Mat& src) {
	if (_pts.size() > 0) {
		for (int i = 0; i < _pts.size(); i++) {
			cv::circle(src, _pts[i], 3, cv::Scalar(255, 0, 255), -1);
		}
		cv::rectangle(src, _rect, cv::Scalar(0, 255, 0), 2);
	}
}


std::vector<cv::Point2f> EsrShape::getAllPts() {
	return _pts;
}

std::vector<cv::Point2f> EsrShape::getPts5() {
	std::vector<cv::Point2f> pts;
	float x = (_pts[27].x + _pts[29].x) / 2;
	float y = (_pts[28].y + _pts[30].y) / 2;
	pts.push_back(cv::Point2f(x, y));

	x = (_pts[31].x + _pts[33].x) / 2;
	y = (_pts[32].y + _pts[34].y) / 2;
	pts.push_back(cv::Point2f(x, y));

	pts.push_back(_pts[65]);
	pts.push_back(_pts[46]);
	pts.push_back(_pts[52]);

	return pts;
}

std::vector<cv::Point2f> EsrShape::filter74to68(const std::vector<cv::Point2f>& pts) {
	std::vector<cv::Point2f> filteredPts;
	for (int i = 0; i < 68; i++) {
		if (i <= 30) {
			filteredPts.push_back(pts[i]);
		}
		else if (i == 31) {
			float x = (pts[27].x + pts[29].x) / 2;
			float y = (pts[28].y + pts[30].y) / 2;
			filteredPts.push_back(cv::Point2f(x, y));
		}
		else if (i <= 35) {
			filteredPts.push_back(pts[i - 1]);
		}
		else if (i == 36) {
			float x = (pts[31].x + pts[33].x) / 2;
			float y = (pts[32].y + pts[34].y) / 2;
			filteredPts.push_back(cv::Point2f(x, y));
		}
		else {
			filteredPts.push_back(pts[i - 2]);
		}
	}
	return filteredPts;
}

std::vector<cv::Point2f> EsrShape::getFilterPts() {
	if (_landnum = 74 && 74 == _pts.size()) {
		std::vector<cv::Point2f> pts = filter74to68(_pts);
		return pts;
	}
	else {
		return _pts;
	}
}