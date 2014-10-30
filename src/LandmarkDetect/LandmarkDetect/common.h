#pragma  once

#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4267)


#include <assert.h>
#include <string>
#include <xstring>
#include <map>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <exception>
#include <cmath>
#include <time.h>
#include <set>
#include <queue>
#include <list>
#include <limits>
#include <fstream>
#include <sstream>
#include <random>
#include <atlstr.h>
#include <atltypes.h>
#include <omp.h>
#include <strstream>
#include <stdarg.h>
#include <string>


#ifdef _DEBUG
#define lnkLIB(name) name "d"
#else
#define lnkLIB(name) name
#endif


#include <opencv2/opencv.hpp> 
#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#define cvLIB(name) lnkLIB("opencv_" name CV_VERSION_ID)

#pragma comment( lib, cvLIB("core"))
#pragma comment( lib, cvLIB("imgproc"))
#pragma comment( lib, cvLIB("highgui"))
#pragma comment( lib, cvLIB("photo"))
