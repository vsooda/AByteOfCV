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
GLuint UITextures[3];


void DrawSongAndDance(GLfloat yRot) {
	static GLfloat vWhite[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	static GLfloat vLightPos[] = { 0.0f, 3.0f, 0.0f, 1.0f };
	M3DVector4f vLightTransformed;
	M3DMatrix44f mCamera;
	modelViewMatrix.GetMatrix(mCamera);
	m3dTransformVector4(vLightTransformed, vLightPos, mCamera);

	modelViewMatrix.PushMatrix();
	modelViewMatrix.Translatev(vLightPos);
	shaderManager.UseStockShader(GLT_SHADER_FLAT,
		transformPipeline.GetModelViewProjectionMatrix(),
		vWhite);
	sphereBatch.Draw();
	modelViewMatrix.PopMatrix();
	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	for (int i = 0; i < num_spheres; i++) {
		modelViewMatrix.PushMatrix();
		modelViewMatrix.MultMatrix(spheres[i]);
		shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
			modelViewMatrix.GetMatrix(),
			transformPipeline.GetProjectionMatrix(),
			vLightTransformed,
			vWhite,
			0);
		sphereBatch.Draw();
		modelViewMatrix.PopMatrix();
	}

	modelViewMatrix.Translate(0.0, 0.2f, -2.5f);
	modelViewMatrix.PushMatrix();
	modelViewMatrix.Rotate(yRot, 0.0f, 1.0f, 0.0f);

	glBindTexture(GL_TEXTURE_2D, UITextures[1]);
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
		modelViewMatrix.GetMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightTransformed,
		vWhite,
		0);

	torusBatch.Draw();
	modelViewMatrix.PopMatrix();
	modelViewMatrix.Rotate(yRot*-2.0f, 0.0f, 1.0f, 0.0f);
	modelViewMatrix.Translate(0.8f, 0.0f, 0.0f);
	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
		modelViewMatrix.GetMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightTransformed,
		vWhite,
		0);
	sphereBatch.Draw();
}


void DrawSongAndDance1(GLfloat yRot)		// Called to draw dancing objects
{
	static GLfloat vWhite[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	static GLfloat vLightPos[] = { 0.0f, 3.0f, 0.0f, 1.0f };

	// Get the light position in eye space
	M3DVector4f	vLightTransformed;
	M3DMatrix44f mCamera;
	modelViewMatrix.GetMatrix(mCamera);
	m3dTransformVector4(vLightTransformed, vLightPos, mCamera);

	// Draw the light source
	modelViewMatrix.PushMatrix();
	modelViewMatrix.Translatev(vLightPos);
	shaderManager.UseStockShader(GLT_SHADER_FLAT,
		transformPipeline.GetModelViewProjectionMatrix(),
		vWhite);
	sphereBatch.Draw();
	modelViewMatrix.PopMatrix();

	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	for (int i = 0; i < num_spheres; i++) {
		modelViewMatrix.PushMatrix();
		modelViewMatrix.MultMatrix(spheres[i]);
		shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
			modelViewMatrix.GetMatrix(),
			transformPipeline.GetProjectionMatrix(),
			vLightTransformed,
			vWhite,
			0);
		sphereBatch.Draw();
		modelViewMatrix.PopMatrix();
	}

	// Song and dance
	modelViewMatrix.Translate(0.0f, 0.2f, -2.5f);
	modelViewMatrix.PushMatrix();	// Saves the translated origin
	modelViewMatrix.Rotate(yRot, 0.0f, 1.0f, 0.0f);

	// Draw stuff relative to the camera
	glBindTexture(GL_TEXTURE_2D, UITextures[1]);
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
		modelViewMatrix.GetMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightTransformed,
		vWhite,
		0);
	torusBatch.Draw();
	modelViewMatrix.PopMatrix(); // Erased the rotate

	modelViewMatrix.Rotate(yRot * -2.0f, 0.0f, 1.0f, 0.0f);
	modelViewMatrix.Translate(0.8f, 0.0f, 0.0f);

	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_POINT_LIGHT_DIFF,
		modelViewMatrix.GetMatrix(),
		transformPipeline.GetProjectionMatrix(),
		vLightTransformed,
		vWhite,
		0);
	sphereBatch.Draw();
}

bool loadTGATexture(const char* szFilename, GLenum minFilter, GLenum magFilter, GLenum wrapMode) {
	GLbyte *pBits;
	int nWidth, nHeight, nComponents;
	GLenum eFormat;
	pBits = gltReadTGABits(szFilename, &nWidth, &nHeight, &nComponents, &eFormat);
	if (pBits == NULL) {
		return false;
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapMode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapMode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB, nWidth, nHeight, 0,
		eFormat, GL_UNSIGNED_BYTE, pBits);
	free(pBits);
	if (minFilter == GL_LINEAR_MIPMAP_LINEAR ||
		minFilter == GL_LINEAR_MIPMAP_NEAREST ||
		minFilter == GL_NEAREST_MIPMAP_LINEAR ||
		minFilter == GL_NEAREST_MIPMAP_NEAREST) {
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	return true;
}


void changeSize(int w, int h) {
	if (h == 0)
		h = 1;
	glViewport(0, 0, w, h);

	transformPipeline.SetMatrixStacks(modelViewMatrix, projectionMatrix);
	viewFrustum.SetPerspective(35.0, float(w) / float(h), 1.0, 100.0);
	projectionMatrix.LoadMatrix(viewFrustum.GetProjectionMatrix());
	modelViewMatrix.LoadIdentity();
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

void SetupRC()
{
	// Make sure OpenGL entry points are set
	glewInit();

	// Initialze Shader Manager
	shaderManager.InitializeStockShaders();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	// This makes a torus
	gltMakeTorus(torusBatch, 0.4f, 0.15f, 40, 20);

	// This makes a sphere
	gltMakeSphere(sphereBatch, 0.1f, 26, 13);


	// Make the solid ground
	GLfloat texSize = 10.0f;
	floorBatch.Begin(GL_TRIANGLE_FAN, 4, 1);
	floorBatch.MultiTexCoord2f(0, 0.0f, 0.0f);
	floorBatch.Vertex3f(-20.0f, -0.41f, 20.0f);

	floorBatch.MultiTexCoord2f(0, texSize, 0.0f);
	floorBatch.Vertex3f(20.0f, -0.41f, 20.0f);

	floorBatch.MultiTexCoord2f(0, texSize, texSize);
	floorBatch.Vertex3f(20.0f, -0.41f, -20.0f);

	floorBatch.MultiTexCoord2f(0, 0.0f, texSize);
	floorBatch.Vertex3f(-20.0f, -0.41f, -20.0f);
	floorBatch.End();

	// Make 3 texture objects
	glGenTextures(3, UITextures);

	// Load the Marble
	glBindTexture(GL_TEXTURE_2D, UITextures[0]);
	loadTGATexture("../data/marble.tga", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_REPEAT);

	// Load Mars
	glBindTexture(GL_TEXTURE_2D, UITextures[1]);
	loadTGATexture("../data/marslike.tga", GL_LINEAR_MIPMAP_LINEAR,
		GL_LINEAR, GL_CLAMP_TO_EDGE);

	// Load Moon
	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	loadTGATexture("../data/moonlike.tga", GL_LINEAR_MIPMAP_LINEAR,
		GL_LINEAR, GL_CLAMP_TO_EDGE);

	// Randomly place the spheres
	for (int i = 0; i < num_spheres; i++) {
		GLfloat x = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		GLfloat z = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		spheres[i].SetOrigin(x, 0.0f, z);
	}
}

void SetupRC1() {
	glewInit();
	shaderManager.InitializeStockShaders();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	gltMakeTorus(torusBatch, 0.4f, 0.15f, 40, 20);
	gltMakeSphere(sphereBatch, 0.1f, 26, 13);
	
	GLfloat texSize = 10.0f;
	floorBatch.Begin(GL_TRIANGLE_FAN, 4, 1);

	floorBatch.MultiTexCoord2f(0, 0.0f, 0.0f);
	floorBatch.Vertex3f(-20.0f, -0.41f, 20.0f);

	floorBatch.MultiTexCoord2f(0, texSize, 0.0f);
	floorBatch.Vertex3f(20.0f, -0.41f, 20.0f);

	floorBatch.MultiTexCoord2f(0, texSize, texSize);
	floorBatch.Vertex3f(20.0f, -0.41f, -20.0f);

	floorBatch.MultiTexCoord2f(0, 0.0f, texSize);
	floorBatch.Vertex3f(-20.0f, -0.41f, -20.0f);

	floorBatch.End();

	glGenTextures(3, UITextures);
	glBindTexture(GL_TEXTURE_2D, UITextures[0]);
	loadTGATexture("../data/marble.tga", GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_REPEAT);
	glBindTexture(GL_TEXTURE_2D, UITextures[1]);
	loadTGATexture("../data/marslike.tga", GL_LINEAR_MIPMAP_LINEAR,
		GL_LINEAR, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, UITextures[2]);
	loadTGATexture("../data/moonlike.tga", GL_LINEAR_MIPMAP_LINEAR,
		GL_LINEAR, GL_CLAMP_TO_EDGE);

	for (int i = 0; i < num_spheres; i++) {
		GLfloat x = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		GLfloat z = ((GLfloat)((rand() % 400) - 200) * 0.1f);
		spheres[i].SetOrigin(x, 0.0f, z);
	}
}

void Render(void) {
	static CStopWatch rotTimer;
	float yRot = rotTimer.GetElapsedSeconds() * 60.0f;
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	modelViewMatrix.PushMatrix();
	M3DMatrix44f mCamera;
	cameraFrame.GetCameraMatrix(mCamera);
	modelViewMatrix.MultMatrix(mCamera);

	modelViewMatrix.PushMatrix();
	modelViewMatrix.Scale(1.0f, -1.0f, 1.0f);
	modelViewMatrix.Translate(0.0f, 0.8f, 0.0f);
	glFrontFace(GL_CW);
	DrawSongAndDance1(yRot);
	glFrontFace(GL_CCW);
	modelViewMatrix.PopMatrix();

	glEnable(GL_BLEND);
	glBindTexture(GL_TEXTURE_2D, UITextures[0]);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	static GLfloat vFloorColor[] = { 1.0f, 1.0f, 1.0f, 0.75f };
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_MODULATE,
		transformPipeline.GetModelViewProjectionMatrix(),
		vFloorColor,
		0);

	floorBatch.Draw();
	glDisable(GL_BLEND);
	DrawSongAndDance1(yRot);
	modelViewMatrix.PopMatrix();
	glutSwapBuffers();
	glutPostRedisplay();
}
void ShutdownRC(void) {
	glDeleteTextures(3, UITextures);
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
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutCreateWindow("sphereworld");
	glutSpecialFunc(specialKeys);
	glutReshapeFunc(changeSize);
	glutDisplayFunc(Render);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		return 1;
	}
	SetupRC1();
	glutMainLoop();
	ShutdownRC();
	return 0;

}