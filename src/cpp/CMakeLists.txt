cmake_minimum_required (VERSION 2.6) 

PROJECT(algocv)
SET(PROJECT_VERSION "0.1")
set(CMAKE_BUILD_TYPE Debug)

set(OpenCV_DIR "/home/sooda/tools/OpenCV/opencv-2.4.8/build")

# Opencv Package
FIND_PACKAGE( OpenCV REQUIRED )
IF (${OpenCV_VERSION} VERSION_LESS 2.3.0)
	MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}")
ENDIF()

SET(SRC
 main.cpp
 photoAlgo.h
 photoUtil.cpp
 epdfilter.cpp
 illumtransform.cpp
 inpaint.cpp
 makeup.cpp
 quilt.cpp
 salancy.cpp
 skindetector.cpp
 tonemapping.cpp
 3rdparty/maxflow/maxflow.cpp
 3rdparty/maxflow/graph.cpp
)


ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC} )
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}  ${OpenCV_LIBS} )

foreach(f ${RESOURCES})
file( COPY ${f} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach(f)
