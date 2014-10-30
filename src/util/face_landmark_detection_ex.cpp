// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    This face detector is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset.  

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <iostream>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            //cout << "Call this program like this:" << endl;
            //cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            //cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            //cout << "http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

//        freopen("1.txt", "w", stdout);

        image_window win;
        // Loop over all the images provided on the command line.
        for (int i = 2; i < argc; ++i)
        {
            cout << "processing image " << argv[i] << endl;
            array2d<rgb_pixel> img;
            load_image(img, argv[i]);
            
            cv::Mat srcGray = cv::imread(argv[i], 0);
            equalizeHist(srcGray, srcGray);
            cv::CascadeClassifier cascade, nestedCascade;
            //char cascadeFilename[] = "haarcascade_profileface.xml";
            char cascadeFilename[] = "haarcascade_frontalface_alt2.xml";
            //char cascadeFilename[] = "haarcascade_frontalface_default.xml";
            cascade.load(cascadeFilename);
            std::vector<cv::Rect> faces;
            //cascade.detectMultiScale(srcGray, faces, 1.1, 2,CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(15, 15) );
            cascade.detectMultiScale(srcGray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100) );
            //int left_x = faces[0].x + 0.05 * faces[0].width;
            //int left_y = faces[0].y + 0.15 * faces[0].height;
            //int right_x = faces[0].x + faces[0].width * 0.9;
            //int right_y = faces[0].y + faces[0].height * 1.1 ;
            int left_x = faces[0].x ;
            int left_y = faces[0].y ;
            int right_x = faces[0].x + faces[0].width;
            int right_y = faces[0].y + faces[0].height;

            //cv::rectangle(srcGray, cv::Point(left_x, left_y), cv::Point(right_x, right_y), cv::Scalar(255, 0, 0));
            //cv::imshow("detection", srcGray);
            //cv::waitKey();

         
            rectangle detection(left_x, left_y, right_x, right_y);
            std::vector<rectangle> dets;
            dets.push_back(detection);
            std::vector<rectangle> dets1 = detector(img);

            // Make the image larger so we can detect small faces.
 //           pyramid_up(img);
 //           std::cout << img.nr() << " "  << img.nc() << std::endl; 

            // Now tell the face detector to give us a list of bounding boxes
            // around all the faces in the image.
            //std::vector<rectangle> dets = detector(img);
            ////cout << "Number of faces detected: " << dets.size() << endl;

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            std::vector<full_object_detection> shapes1;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]);
                //cout << "number of parts: "<< shape.num_parts() << endl;
                //cout << "pixel position of first part:  " << shape.part(0) << endl;
                //cout << "pixel position of second part: " << shape.part(1) << endl;
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);
            }
            for (unsigned long j = 0; j < dets1.size(); ++j)
            {
                full_object_detection shape = sp(img, dets1[j]);
                shapes1.push_back(shape);
            }

            full_object_detection res = shapes[0];
            full_object_detection res1 = shapes1[0];
            //for(int i = 0; i < 68; i++) {
            //    std::cout << res.part(i) << std::endl;
            cv::Mat dst = cv::imread(argv[i]);
            //FILE *fout = fopen("0.lm", "w");
            for(int i = 0; i < 68; i++) {
                point pt = res.part(i);
                int tempx = pt.x();
                int tempy = pt.y();
                //fprintf(fout, "%.2f,%.2f\n", (float)tempx, (float)tempy);
                cv::circle(dst, cv::Point(tempx, tempy), 1, cv::Scalar(255, 255, 0));
                point pt1 = res1.part(i);
                int tempx1 = pt1.x();
                int tempy1 = pt1.y();
                cv::circle(dst, cv::Point(tempx1, tempy1), 1, cv::Scalar(255, 255, 255));
            }
            cv::rectangle(dst, cv::Point(left_x, left_y), cv::Point(right_x, right_y), cv::Scalar(255, 0, 255));
            cv::rectangle(dst, cv::Point(dets1[0].left(), dets1[0].top()), cv::Point(dets1[0].right(), dets1[0].bottom()), cv::Scalar(255, 0, 0));
            cv::imshow("dst", dst);
            cv::waitKey();
            //cout << "Hit enter to process the next image..." << endl;
        }
    }
    catch (exception& e)
    {
        //cout << "\nexception thrown!" << endl;
        //cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

