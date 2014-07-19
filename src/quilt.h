#ifndef __OPENCV_QUILT_H__ 
#define __OPENCV_QUILT_H__

//#include "cvconfig.h"
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgproc/imgproc_c.h"
#include <vector>
#include <iostream>

namespace customCV {
	void quilt(cv::Mat& texture, cv::Mat& dst, int width, int niter);
}

#endif