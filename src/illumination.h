#ifndef __OPENCV_ILLUMINATION_H__ 
#define __OPENCV_ILLUMINATION_H__

#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/imgproc/imgproc_c.h"
#include <vector>
#include <iostream>
#include <stdarg.h>
#include "util.h"

namespace customCV {
	void illumProcess(char * src1, char* src2, char* save_name, char* mask_name, customCV::funcParam fp);
	void transform_test(char* src_filename, char* dst_filename, char* save_name, char* mask_name);
	cv::Mat doTransform(char* src_filename, char* dst_filename, char* save_name, char* mask_name);


}


#endif