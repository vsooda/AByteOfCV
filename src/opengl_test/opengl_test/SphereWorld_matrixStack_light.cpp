/*
使用变换管线、矩阵堆栈实现物体，旋转，光源
*/
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

const int num_spheres = 50;
GLFrame spheres[num_spheres];

//global view (frustum: 平头截体)
GLFrustum viewFrustum;//投影相关
GLShaderManager shaderManager;
GLTriangleBatch torusBatch;
GLTriangleBatch sphereBatch;
GLMatrixStack modelViewMatrix;
GLMatrixStack projectionMatrix;
GLGeometryTransform transformPipeline;
GLBatch floorBatch;
GLFrame cameraFrame;//实现了相机类（效果类似于欧拉角，四元数）


void changeSize(int w, int h) {
	if (h == 0)
		h = 1;
	//viewport 的设置是最后光栅化水平，竖直方向的平移。同时设置宽度高度
	//这里设置的是成像之后到光栅化之间的处理。
	glViewport(0, 0, w, h);
	//透视投影计算。 第一个参数越小，视图越大
	//为了想要的坐标系设置一个透视投影矩阵. 将其映射到单位立方体内。裁剪坐标
	//而摄像头的设置属于成像环节，需要在那部分进行设置投影矩阵等
	viewFrustum.SetPerspective(35.0, float(w) / float(h), 1.0, 100.0);
	projectionMatrix.LoadMatrix(viewFrustum.GetProjectionMatrix());
	transformPipeline.SetMatrixStacks(modelViewMatrix, projectionMatrix);
}

void renderScene(void) {
	static GLfloat vFloorColor[] = { 0.0f, 1.0f, 0.0f, 1.0f };
	static GLfloat vTorusColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	static GLfloat vSphereColor[] = { 0.0f, 0.0f, 1.0f, 1.0f };

	static CStopWatch rotTimer;
	//基于时间的动画。因为不能因为不同的帧率导致物体的转动速度居然不一致。
	float yRot = rotTimer.GetElapsedSeconds() * 60.0f;
	//清楚颜色缓冲区和深度缓冲区
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	modelViewMatrix.PushMatrix();
	M3DMatrix44f mCamera;
	cameraFrame.GetCameraMatrix(mCamera);
	modelViewMatrix.PushMatrix(mCamera);


	M3DVector4f vLightPos = { 0.0f, 10.0f, 5.0f, 1.0f };
	M3DVector4f vLightEyePos;
	m3dTransformVector4(vLightEyePos, vLightPos, mCamera);

	shaderManager.UseStockShader(GLT_SHADER_FLAT,
		transformPipeline.GetModelViewProjectionMatrix(),
		vFloorColor);
	floorBatch.Draw();

	for (int i = 0; i < num_spheres; i++) {
		modelViewMatrix.PushMatrix();
		modelViewMatrix.MultMatrix(spheres[i]);
		//shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(),
		//	vSphereColor);
		shaderManager.UseStockShader(GLT_SHADER_POINT_LIGHT_DIFF, transformPipeline.GetModelViewMatrix(),
			transformPipeline.GetProjectionMatrix(),
			vLightEyePos, vSphereColor);
		sphereBatch.Draw();
		modelViewMatrix.PopMatrix();
	}


	modelViewMatrix.Translate(0.0f, 0.0f, -2.5f);
	//保存平移
	modelViewMatrix.PushMatrix();
	modelViewMatrix.Rotate(yRot, 0.0f, 1.0f, 1.0f);
	//shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(),
	//	vTorusColor);
	shaderManager.UseStockShader(GLT_SHADER_POINT_LIGHT_DIFF, transformPipeline.GetModelViewMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightEyePos, vTorusColor);
	torusBatch.Draw();
	modelViewMatrix.PopMatrix(); //"消除以前的旋转

	//应用另一个旋转
	modelViewMatrix.Rotate(yRot * -2.0f, 0.0f, 1.0f, 0.0f);
	modelViewMatrix.Translate(0.1f, 0.0f, 0.0f);
	//shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(),
	//	vSphereColor);
	shaderManager.UseStockShader(GLT_SHADER_POINT_LIGHT_DIFF, transformPipeline.GetModelViewMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightEyePos, vSphereColor);
	sphereBatch.Draw();

	//出栈，保存之前的模型视图矩阵（单位矩阵）
	modelViewMatrix.PopMatrix();
	modelViewMatrix.PopMatrix();
	glutSwapBuffers();
	glutPostRedisplay();
}

void SetupRC() {
	shaderManager.InitializeStockShaders();
	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//双面
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	gltMakeTorus(torusBatch, 0.4f, 0.15f, 30, 30);
	gltMakeSphere(sphereBatch, 0.1f, 26, 13);
	//绘画地板绿色网格
	//324表示GL_LINES的数目
	floorBatch.Begin(GL_LINES, 324);
	for (GLfloat x = -20.0; x <= 20.0f; x += 0.5) {
		floorBatch.Vertex3f(x, -0.55f, 20.0f);
		floorBatch.Vertex3f(x, -0.55f, -20.0f);
		floorBatch.Vertex3f(20.0f, -0.55f, x);
		floorBatch.Vertex3f(-20.0f, -0.55f, x);
	}
	floorBatch.End();
	for (int i = 0; i < num_spheres; i++) {
		GLfloat x = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		GLfloat z = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		spheres[i].SetOrigin(x, 0.0f, z);
	}
}

void specialKeys(int key, int k, int y) {
	float linear = 0.1f;
	float angular = float(m3dDegToRad(0.5f));
	if (key == GLUT_KEY_UP)
		cameraFrame.MoveForward(linear);
	if (key == GLUT_KEY_DOWN)
		cameraFrame.MoveForward(-linear);
	if (key == GLUT_KEY_LEFT)
		cameraFrame.RotateWorld(angular, 0.0f, 1.0f, 0.0f);
	if (key == GLUT_KEY_RIGHT)
		cameraFrame.RotateWorld(-angular, 0.0f, 1.0f, 0.0f);
}

int main(int argc, char* argv[])
{
	gltSetWorkingDirectory(argv[0]);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB  | GLUT_DEPTH |
		GLUT_STENCIL);
	glutInitWindowSize(800, 600);
	glutCreateWindow("sphereworld");
	glutSpecialFunc(specialKeys);
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