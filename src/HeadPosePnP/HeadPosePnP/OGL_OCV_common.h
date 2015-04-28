/*
 *  OGL_OCV_common.h
 *  HeadReplacement
 *
 *  Created by Roy Shilkrot on 11/21/11.
 *  Copyright 2011 MIT. All rights reserved.
 *
 */

#include <opencv2/opencv.hpp>

using namespace cv;

#include <GL/glew.h>
#ifdef __APPLE__
#include <glut/glut.h>
#else
#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif


void copyImgToTex(const Mat& _tex_img, GLuint* texID, double* _twr, double* _thr);

typedef struct my_texture {
	GLuint tex_id;
	double twr,thr,aspect_w2h;
	Mat image;
	my_texture():tex_id(-1),twr(1.0),thr(1.0) {}
	bool initialized;
	void set(const Mat& ocvimg) { 
		ocvimg.copyTo(image); 
		copyImgToTex(image, &tex_id, &twr, &thr); 
		aspect_w2h = (double)ocvimg.cols/(double)ocvimg.rows;
	}
} OpenCVGLTexture;

void glEnable2D();	// setup 2D drawing
void glDisable2D(); // end 2D drawing
OpenCVGLTexture MakeOpenCVGLTexture(const Mat& _tex_img); // create an OpenCV-OpenGL image
void drawOpenCVImageInGL(const OpenCVGLTexture& tex); // draw an OpenCV-OpenGL image
