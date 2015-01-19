/*
ʹ�ñ任���ߡ������ջʵ�����壬��ת����Դ
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

//global view (frustum: ƽͷ����)
GLFrustum viewFrustum;//ͶӰ���
GLShaderManager shaderManager;
GLTriangleBatch torusBatch;
GLTriangleBatch sphereBatch;
GLMatrixStack modelViewMatrix;
GLMatrixStack projectionMatrix;
GLGeometryTransform transformPipeline;
GLBatch floorBatch;
GLFrame cameraFrame;//ʵ��������ࣨЧ��������ŷ���ǣ���Ԫ����


void changeSize(int w, int h) {
	if (h == 0)
		h = 1;
	//viewport ������������դ��ˮƽ����ֱ�����ƽ�ơ�ͬʱ���ÿ�ȸ߶�
	//�������õ��ǳ���֮�󵽹�դ��֮��Ĵ���
	glViewport(0, 0, w, h);
	//͸��ͶӰ���㡣 ��һ������ԽС����ͼԽ��
	//Ϊ����Ҫ������ϵ����һ��͸��ͶӰ����. ����ӳ�䵽��λ�������ڡ��ü�����
	//������ͷ���������ڳ��񻷽ڣ���Ҫ���ǲ��ֽ�������ͶӰ�����
	viewFrustum.SetPerspective(35.0, float(w) / float(h), 1.0, 100.0);
	projectionMatrix.LoadMatrix(viewFrustum.GetProjectionMatrix());
	transformPipeline.SetMatrixStacks(modelViewMatrix, projectionMatrix);
}

void renderScene(void) {
	static GLfloat vFloorColor[] = { 0.0f, 1.0f, 0.0f, 1.0f };
	static GLfloat vTorusColor[] = { 1.0f, 0.0f, 0.0f, 1.0f };
	static GLfloat vSphereColor[] = { 0.0f, 0.0f, 1.0f, 1.0f };

	static CStopWatch rotTimer;
	//����ʱ��Ķ�������Ϊ������Ϊ��ͬ��֡�ʵ��������ת���ٶȾ�Ȼ��һ�¡�
	float yRot = rotTimer.GetElapsedSeconds() * 60.0f;
	//�����ɫ����������Ȼ�����
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
	//����ƽ��
	modelViewMatrix.PushMatrix();
	modelViewMatrix.Rotate(yRot, 0.0f, 1.0f, 1.0f);
	//shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(),
	//	vTorusColor);
	shaderManager.UseStockShader(GLT_SHADER_POINT_LIGHT_DIFF, transformPipeline.GetModelViewMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightEyePos, vTorusColor);
	torusBatch.Draw();
	modelViewMatrix.PopMatrix(); //"������ǰ����ת

	//Ӧ����һ����ת
	modelViewMatrix.Rotate(yRot * -2.0f, 0.0f, 1.0f, 0.0f);
	modelViewMatrix.Translate(0.1f, 0.0f, 0.0f);
	//shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(),
	//	vSphereColor);
	shaderManager.UseStockShader(GLT_SHADER_POINT_LIGHT_DIFF, transformPipeline.GetModelViewMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightEyePos, vSphereColor);
	sphereBatch.Draw();

	//��ջ������֮ǰ��ģ����ͼ���󣨵�λ����
	modelViewMatrix.PopMatrix();
	modelViewMatrix.PopMatrix();
	glutSwapBuffers();
	glutPostRedisplay();
}

void SetupRC() {
	shaderManager.InitializeStockShaders();
	glEnable(GL_DEPTH_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//˫��
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	gltMakeTorus(torusBatch, 0.4f, 0.15f, 30, 30);
	gltMakeSphere(sphereBatch, 0.1f, 26, 13);
	//�滭�ذ���ɫ����
	//324��ʾGL_LINES����Ŀ
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