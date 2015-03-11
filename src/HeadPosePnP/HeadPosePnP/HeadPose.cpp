// HeadPose.cpp : Defines the entry point for the console application.
//


#include "cv.h"
#include "highgui.h"
#include "util.h"
#include "ESR_shape.h"

using namespace cv;

#include <vector>
#include <iostream>
#include <fstream>

using namespace std;

#include <GL/glew.h>
#ifdef __APPLE__
#include <glut/glut.h>
#else
#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif


#include "glm.h"
#include "OGL_OCV_common.h"

void loadNext();
void loadWithPoints(Mat& ip, Mat& img);
void detectNext();

const GLfloat light_ambient[]  = { 0.0f, 0.0f, 0.0f, 1.0f };
const GLfloat light_diffuse[]  = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat light_position[] = { 0.0f, 0.0f, 1.0f, 0.0f };

const GLfloat mat_ambient[]    = { 0.7f, 0.7f, 0.7f, 1.0f };
const GLfloat mat_diffuse[]    = { 0.8f, 0.8f, 0.8f, 1.0f };
const GLfloat mat_specular[]   = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat high_shininess[] = { 100.0f };

double rot[9] = {0};
GLuint textureID;
Mat backPxls;
vector<double> rv(3), tv(3);
Mat rvec(rv),tvec(tv);
Mat camMatrix;

OpenCVGLTexture imgTex,imgWithDrawing;

GLMmodel* head_obj;

std::vector<std::string> names;
std::string dir;
EsrShape *g_esp;

void saveOpenGLBuffer() {
	static unsigned int opengl_buffer_num = 0;
	
	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
	Mat_<Vec3b> opengl_image(vPort[3],vPort[2]);
	{
		Mat_<Vec4b> opengl_image_4b(vPort[3],vPort[2]);
		glReadPixels(0, 0, vPort[2], vPort[3], GL_BGRA, GL_UNSIGNED_BYTE, opengl_image_4b.data);
		flip(opengl_image_4b,opengl_image_4b,0);
		mixChannels(&opengl_image_4b, 1, &opengl_image, 1, &(Vec6i(0,0,1,1,2,2)[0]), 3);
	}
	stringstream ss; ss << "opengl_buffer_" << opengl_buffer_num++ << ".jpg";
	imwrite(ss.str(), opengl_image);
}


void resize(int width, int height)
{
    const float ar = (float) width / (float) height;

    glViewport(0, 0, width, height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glFrustum(-ar, ar, -1.0, 1.0, 2.0, 100.0);
	gluPerspective(47,1.0,0.01,1000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

int __w=250,__h=250;

void key(unsigned char key, int x, int y)
{

    switch (key)
    {
    case 27 :
    case 'Q':
    case 'q': 
		exit(0);
		break;
	case 'w':
	case 'W':
		__w++;
		__w = __w%251;
		break;
	case 'h':
	case 'H':
		__h++;
		__h = __h%250;
		break;
	case ' ':
			saveOpenGLBuffer();
		//loadNext();
		detectNext();
		break;
    default:
        break;
    }

    glutPostRedisplay();
}

void idle(void)
{
    glutPostRedisplay();
}



void myGLinit() {
//    glutSetOption ( GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION ) ;

	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	

	glShadeModel(GL_SMOOTH);

    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glEnable(GL_LIGHT0);
    glEnable(GL_NORMALIZE);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );

    glLightfv(GL_LIGHT0, GL_AMBIENT,  light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE,  light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
    glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse);
    glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
    glMaterialfv(GL_FRONT, GL_SHININESS, high_shininess);

	glEnable(GL_LIGHTING);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
}

void drawAxes() {
	
	//Z = red
	glPushMatrix();
	glRotated(180,0,1,0);
	glColor4d(1,0,0,0.5);
//	glutSolidCylinder(0.05,1,15,20);
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
    glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	glutSolidTetrahedron();
	glPopMatrix();
	
	//Y = green
	glPushMatrix();
	glRotated(-90,1,0,0);
	glColor4d(0,1,0,0.5);
//	glutSolidCylinder(0.05,1,15,20);
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
    glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	glutSolidTetrahedron();
	glPopMatrix();
	
	//X = blue
	glPushMatrix();
	glRotated(-90,0,1,0);
	glColor4d(0,0,1,0.5);
//	glutSolidCylinder(0.05,1,15,20);
    glBegin(GL_LINES);
    glVertex3d(0, 0, 0); glVertex3d(0, 0, 1);
    glEnd();
	glTranslated(0,0,1);
	glScaled(.1,.1,.1);
	glutSolidTetrahedron();
	glPopMatrix();
}	

void display(void)
{	
	// draw the image in the back
	int vPort[4]; glGetIntegerv(GL_VIEWPORT, vPort);
	glEnable2D();
	drawOpenCVImageInGL(imgTex);
	glTranslated(vPort[2]/2.0, 0, 0);
	drawOpenCVImageInGL(imgWithDrawing);
	glDisable2D();

	glClear(GL_DEPTH_BUFFER_BIT); // we want to draw stuff over the image
	
	// draw only on left part
	glViewport(0, 0, vPort[2]/2, vPort[3]);
	
	glPushMatrix();
	
	gluLookAt(0,0,0,0,0,1,0,-1,0);

	// put the object in the right position in space
	Vec3d tvv(tv[0],tv[1],tv[2]);
	glTranslated(tvv[0], tvv[1], tvv[2]);

	// rotate it
	double _d[16] = {	rot[0],rot[1],rot[2],0,
						rot[3],rot[4],rot[5],0,
						rot[6],rot[7],rot[8],0,
						0,	   0,	  0		,1};
	glMultMatrixd(_d);
	
	// draw the 3D head model
	glColor4f(1, 1, 1,0.75);
	glmDraw(head_obj, GLM_SMOOTH);

	//----------Axes
	glScaled(50, 50, 50);
	drawAxes();
	//----------End axes

	glPopMatrix();
	
	// restore to looking at complete viewport
	glViewport(0, 0, vPort[2], vPort[3]); 

	glutSwapBuffers();
}

void init_opengl(int argc,char** argv) {
	glutInitWindowSize(500,250);
    glutInitWindowPosition(40,40);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH); // | GLUT_MULTISAMPLE
    glutCreateWindow("head pose");
	
	myGLinit();
	
    glutReshapeFunc(resize);
    glutDisplayFunc(display);
    glutKeyboardFunc(key);
    //glutSpecialFunc(special);
    glutIdleFunc(idle);
}	

int start_opengl() {

    glutMainLoop();

	return 1;
}

Mat op;

void loadNext() {
	static int counter = 1;
	
	printf("load %d\n",counter);
    
    const char* workingDir = "./";

	char buf[256] = {0};
	sprintf(buf,"%sAngelina_Jolie/Angelina_Jolie_%04d.txt",workingDir,counter);

	vector<Point2f > imagePoints;
	ifstream inputfile(buf);
	if(inputfile.fail()) { 
		cerr << "can't read " << buf << endl; return; 
	}

	for(int i=0;i<7;i++) {
		int x=0,y=0;
		inputfile >> skipws >> x >> y;
		imagePoints.push_back(Point2f((float)x,(float)y));
	}
	inputfile.close();

	Mat ip(imagePoints);
	
	sprintf(buf,"%sAngelina_Jolie/Angelina_Jolie_%04d.jpg",workingDir,counter);

	Mat img = imread(buf);

	imgTex.set(img); //TODO: what if different size??

	// paint 2D feature points
	for(unsigned int i=0;i<imagePoints.size();i++) circle(img,imagePoints[i],2,Scalar(255,0,255),CV_FILLED);

//	cv::imshow("img", img);
//	cv::waitKey();

	loadWithPoints(ip,img);
	
	imgWithDrawing.set(img);
	
	counter = (counter+1);
}

void loadWithPoints(Mat& ip, Mat& img) {
	int max_d = MAX(img.rows,img.cols);
	camMatrix = (Mat_<double>(3,3) << max_d, 0, img.cols/2.0,
										0,	max_d, img.rows/2.0,
										0,	0,	1.0);
	cout << "using cam matrix " << endl << camMatrix << endl;
	
	double _dc[] = {0,0,0,0};
	std::cout << ip.size() << " " << ip.type() << std::endl;
	std::cout << "op: " << op << std::endl;
	std::cout << "ip: " << ip << std::endl;
	std::cout << rvec << std::endl;
	std::cout << tvec << std::endl;

	/*for (int i = 0; i < ip.rows; i++) {
		std::cout << ip.at<cv::Point2f>(i, 0).x << std::endl;
	}*/
	transpose(ip, ip);
	//transpose(op, op);
	solvePnP(op,ip,camMatrix,Mat(1,4,CV_64FC1,_dc),rvec,tvec,false,CV_EPNP);
	//solvePnP(op, ip, camMatrix, Mat(1, 4, CV_64FC1, _dc), rvec, tvec, false, CV_P3P);
	//transpose(ip, ip);
	//transpose(op, op);

	std::cout << "after pnp: rvec " << rvec << std::endl;

	Mat rotM(3,3,CV_64FC1,rot);
	Rodrigues(rvec,rotM);
	double* _r = rotM.ptr<double>();
	printf("rotation mat: \n %.3f %.3f %.3f\n%.3f %.3f %.3f\n%.3f %.3f %.3f\n",
		_r[0],_r[1],_r[2],_r[3],_r[4],_r[5],_r[6],_r[7],_r[8]);

	printf("trans vec: \n %.3f %.3f %.3f\n",tv[0],tv[1],tv[2]);

	double _pm[12] = {_r[0],_r[1],_r[2],tv[0],
					  _r[3],_r[4],_r[5],tv[1],
					  _r[6],_r[7],_r[8],tv[2]};

	Matx34d P(_pm);
	Mat KP = camMatrix * Mat(P);
//	cout << "KP " << endl << KP << endl;

	//reproject object points - check validity of found projection matrix
	for (int i=0; i<op.rows; i++) {
		Mat_<double> X = (Mat_<double>(4,1) << op.at<float>(i,0),op.at<float>(i,1),op.at<float>(i,2),1.0);
//		cout << "object point " << X << endl;
		Mat_<double> opt_p = KP * X;
		Point2f opt_p_img(opt_p(0)/opt_p(2),opt_p(1)/opt_p(2));
//		cout << "object point reproj " << opt_p_img << endl; 
		
		circle(img, opt_p_img, 4, Scalar(0,0,255), 1);
	}
	rotM = rotM.t();// transpose to conform with majorness of opengl matrix
}

void detectNext(EsrShape& esp, const char* filename) {
	vector<Point2f > imagePoints;

	Mat img = imread(filename);

	esp.detect(img);
	imagePoints = esp.getAllPts();
	//imagePoints = esp.getPts5();
	Mat ip(imagePoints);
	std::cout << ip.size() << std::endl;

	imgTex.set(img); //TODO: what if different size??

	// paint 2D feature points
	for (unsigned int i = 0; i < imagePoints.size(); i++) circle(img, imagePoints[i], 2, Scalar(255, 0, 255), CV_FILLED);

	loadWithPoints(ip, img);
	imgWithDrawing.set(img);
}


void detectNext() {
	static int count = 1;
	vector<Point2f > imagePoints;
	std::string filename = dir + names[count];
	Mat img = imread(filename.c_str());
	std::cout << count << " " << filename << std::endl;

	g_esp->detect(img);
	imagePoints = g_esp->getAllPts();
	//imagePoints = esp.getPts5();
	Mat ip(imagePoints);
	std::cout << ip.size() << std::endl;

	imgTex.set(img); //TODO: what if different size??

	// paint 2D feature points
	for (unsigned int i = 0; i < imagePoints.size(); i++) circle(img, imagePoints[i], 2, Scalar(255, 0, 255), CV_FILLED);

	loadWithPoints(ip, img);
	imgWithDrawing.set(img);
	count++;
}

void jolie_test(int argc, char** argv) {
	vector<Point3f > modelPoints;
	modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
	modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
	modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
	modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
	modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695)
	modelPoints.push_back(Point3f(-61.8886, 127.797, -89.4523));	// l ear (v 2011)
	modelPoints.push_back(Point3f(127.603, 126.9, -83.9129));		// r ear (v 1138)

	head_obj = glmReadOBJ("head-obj.obj");
	//head_obj = glmReadOBJ("b3.obj");

	op = Mat(modelPoints);
	transpose(op, op);
	Scalar m = mean(Mat(modelPoints));

	cout << "model points " << op << endl;

	rvec = Mat(rv);
	double _d[9] = { 1, 0, 0,
		0, -1, 0,
		0, 0, -1 };
	Rodrigues(Mat(3, 3, CV_64FC1, _d), rvec);
	tv[0] = 0; tv[1] = 0; tv[2] = 1;
	tvec = Mat(tv);

	camMatrix = Mat(3, 3, CV_64FC1);

	init_opengl(argc, argv); // get GL context, for loading textures

	// prepare OpenCV-OpenGL images
	imgTex = MakeOpenCVGLTexture(Mat());
	imgWithDrawing = MakeOpenCVGLTexture(Mat());
	
	loadNext();

	start_opengl();
}

void dlib_test() {
	EsrShape esp("D:/data/front_face.dat", "D:/data/sp_10000.dat");
	std::vector<std::string> names;
	std::string dir;
	int cnt = readDir("D:/data/face/*.jpg", names, dir);
	for (int i = 0; i < cnt; i++) {
		std::string filename = dir + names[i];
		cv::Mat img = cv::imread(filename.c_str());
		//cv::resize(img, img, cv::Size(500, 500));

		cv::Mat dst = img.clone();

		bool detectFlag = esp.detect(img);
		if (!detectFlag) {
			std::cout << filename << " no detect " << std::endl;
			cv::imshow("dd", dst);
			cv::waitKey();
			continue;
		}
		//getAllPtsgetAllPts();
		std::vector<cv::Point2f> pts = esp.getFilterPts();
		//std::vector<cv::Point2f> pts = esp.getAllPts();
		for (int i = 0; i < pts.size(); i++) {
			cv::circle(dst, pts[i], 3, cv::Scalar(255, 255, 255), -1);
		}
		cv::imshow("dd", dst);
		cv::waitKey();
		continue;
	}
}

void test(int argc, char** argv) {
	vector<Point3f > modelPoints;
	cv::Mat model = readPlyData("D:/data/b33.ply");
	std::cout << model.size() << std::endl;
	cv::Mat selectModel = selectPlyData(model, "D:/data/pt.txt");
	std::cout << selectModel << std::endl;

	op = selectModel;

	modelPoints.push_back(Point3f(2.37427, 110.322, 21.7776));	// l eye (v 314)
	modelPoints.push_back(Point3f(70.0602, 109.898, 20.8234));	// r eye (v 0)
	modelPoints.push_back(Point3f(36.8301, 78.3185, 52.0345));	//nose (v 1879)
	modelPoints.push_back(Point3f(14.8498, 51.0115, 30.2378));	// l mouth (v 1502)
	modelPoints.push_back(Point3f(58.1825, 51.0115, 29.6224));	// r mouth (v 695)
	//modelPoints.push_back(Point3f(-61.8886, 127.797, -89.4523));	// l ear (v 2011)
	//modelPoints.push_back(Point3f(127.603, 126.9, -83.9129));		// r ear (v 1138)
	//op = Mat(modelPoints);

	//transpose(op, op);

	//head_obj = glmReadOBJ("head-obj.obj");
	head_obj = glmReadOBJ("D:/data/b33.obj");

	Scalar m = mean(Mat(modelPoints));

	cout << "model points " << op << endl;

	rvec = Mat(rv);
	double _d[9] = { 1, 0, 0,
		0, -1, 0,
		0, 0, -1 };
	Rodrigues(Mat(3, 3, CV_64FC1, _d), rvec);
	tv[0] = 0; tv[1] = 0; tv[2] = 1;
	tvec = Mat(tv);

	camMatrix = Mat(3, 3, CV_64FC1);

	init_opengl(argc, argv); // get GL context, for loading textures

	// prepare OpenCV-OpenGL images
	imgTex = MakeOpenCVGLTexture(Mat());
	imgWithDrawing = MakeOpenCVGLTexture(Mat());

	char filename[] = "D:/data/face/img_1227.jpg";
//	char filename[] = "Angelina_Jolie/Angelina_Jolie_0001.jpg";
	//int cnt = readDir("D:/data/face/*.jpg", names, dir);
	int cnt = readDir("Angelina_Jolie/*.jpg", names, dir);

	//detectNext(esp, filename);
	detectNext();
	

	start_opengl();
}


int main(int argc, char** argv)
{
	g_esp = new EsrShape("D:/data/front_face.dat", "D:/data/sp_10000.dat");
	test(argc, argv);
	//jolie_test(argc, argv);
//	dlib_test();
	return 0;
}


