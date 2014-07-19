#ifndef __OPENCV_INPAINT_H__ 
#define __OPENCV_INPAINT_H__

//#include "cvconfig.h"
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/photo/photo.hpp"

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/photo/photo_tegra.hpp"
#endif

namespace customCV {
	void cvInpaint( const CvArr* _input_img, const CvArr* _inpaint_mask, CvArr* _output_img,
           double inpaintRange, int flags );
	void inpaint(cv::InputArray _src, cv::InputArray _mask, cv::OutputArray _dst,
                  double inpaintRange, int flags );

}

#endif