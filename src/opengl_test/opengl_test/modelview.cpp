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

//global view (frustum: ƽͷ����)
GLFrustum viewFrustum;//ͶӰ���

GLShaderManager shaderManager;
GLTriangleBatch torusBatch;
GLFrame cameraFrame;

void changeSize(int w, int h) {
	if (h == 0)
		h = 1;
	//viewport ������������դ��ˮƽ����ֱ�����ƽ�ơ�ͬʱ���ÿ�ȸ߶�
	//�������õ��ǳ���֮�󵽹�դ��֮��Ĵ���
	glViewport(100, 0, w, h);
	//͸��ͶӰ���㡣 ��һ������ԽС����ͼԽ��
	//Ϊ����Ҫ������ϵ����һ��͸��ͶӰ����. ����ӳ�䵽��λ�������ڡ��ü�����
	//������ͷ���������ڳ��񻷽ڣ���Ҫ���ǲ��ֽ�������ͶӰ�����
	viewFrustum.SetPerspective(35.0, float(w) / float(h), 1.0, 100.0);
}

void renderScene(void) {
	static CStopWatch rotTimer;
	float yRot = rotTimer.GetElapsedSeconds() * 60.0f;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	M3DMatrix44f mTranslate, mRotate, mModelview, mModelViewProjection;
	//�����д�ԭ��λ�ã��۲�λ�ã��ƿ����������ǲ��ܿ������С�����z -2.5����������ϵ��ʾ����Ļ�ڲ��ƶ�2.5
	m3dTranslationMatrix44(mTranslate, 0.0f, 0.0f, -2.5f);
	//����ƽ�ƾ���Ϊ��1 0 0 0 0 1 0 0 0 0 1 0 0 0 -2.5 1
	//m3dRotationMatrix44�е�xyz������ת��
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
	//ƽ����ɫ�������þ���ʹ���ṩ�ľ���Զ������ת��
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
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//˫��
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