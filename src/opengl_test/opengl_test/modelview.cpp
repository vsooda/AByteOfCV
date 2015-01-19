#include <GLTools.h>	// OpenGL toolkit
#include <GLMatrixStack.h>
#include <GLFrame.h>
#include <GLFrustum.h>
#include <GLGeometryTransform.h>
#include <GLBatch.h>
#include <StopWatch.h>

#include <math.h>
#ifdef __APPLE__
#include <glut/glut.h>
#else
#define FREEGLUT_STATIC
#include <GL/glut.h>
#endif

#include <iostream>
using namespace std;

//global view (frustum: 平头截体)
GLFrustum viewFrustum;//投影相关

GLShaderManager shaderManager;
GLTriangleBatch torusBatch;
GLFrame cameraFrame;

void changeSize(int w, int h) {
	if (h == 0)
		h = 1;
	//viewport 的设置是最后光栅化水平，竖直方向的平移。同时设置宽度高度
	//这里设置的是成像之后到光栅化之间的处理。
	glViewport(100, 0, w, h);
	//透视投影计算。 第一个参数越小，视图越大
	//为了想要的坐标系设置一个透视投影矩阵. 将其映射到单位立方体内。裁剪坐标
	//而摄像头的设置属于成像环节，需要在那部分进行设置投影矩阵等
	viewFrustum.SetPerspective(35.0, float(w) / float(h), 1.0, 100.0);
}

void renderScene(void) {
	static CStopWatch rotTimer;
	float yRot = rotTimer.GetElapsedSeconds() * 60.0f;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	M3DMatrix44f mTranslate, mRotate, mModelview, mModelViewProjection;
	//讲花托从原点位置（观察位置）移开。这样我们才能看到花托。这里z -2.5在右手坐标系表示向屏幕内部移动2.5
	m3dTranslationMatrix44(mTranslate, 0.0f, 0.0f, -2.5f);
	//这里平移矩阵为：1 0 0 0 0 1 0 0 0 0 1 0 0 0 -2.5 1
	//m3dRotationMatrix44中的xyz设置旋转轴
	m3dRotationMatrix44(mRotate, m3dDegToRad(yRot), 1.0f, 1.0f, 1.0f);
	//-0.962353 0 -0.271802 0 0 1 0 0 0.271802 0 -0.962353 0 0 0 0 1
	//for (int i = 0; i < 16; i++) {
	//	printf("%.2f ", mRotate[i]);
	//}
	//cout << endl;
	m3dMatrixMultiply44(mModelview, mTranslate, mRotate);
	m3dMatrixMultiply44(mModelViewProjection, viewFrustum.GetProjectionMatrix(),
		mModelview);
	GLfloat vBlack[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	//平面着色器的作用就是使用提供的矩阵对顶点进行转换
	shaderManager.UseStockShader(GLT_SHADER_FLAT, mModelViewProjection, vBlack);
	torusBatch.Draw();
	glutSwapBuffers();
	glutPostRedisplay();
}

void SetupRC() {
	glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	shaderManager.InitializeStockShaders();
	gltMakeTorus(torusBatch, 0.4f, 0.15f, 30, 30);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//双面
}

int main(int argc, char* argv[])
{
	gltSetWorkingDirectory(argv[0]);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH |
		GLUT_STENCIL);
	glutInitWindowSize(600, 800);
	glutCreateWindow("ModelViewProjection example");
	glutReshapeFunc(changeSize);
	glutDisplayFunc(renderScene);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		return 1;
	}
	SetupRC();
	glutMainLoop();
	return 0;

}