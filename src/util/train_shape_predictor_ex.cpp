// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to use dlib's implementation of the paper:
        One Millisecond Face Alignment with an Ensemble of Regression Trees by
        Vahid Kazemi and Josephine Sullivan, CVPR 2014
    
    In particular, we will train a face landmarking model based on a small dataset 
    and then evaluate it.  If you want to visualize the output of the trained
    model on some images then you can run the face_landmark_detection_ex.cpp
    example program with sp.dat as the input model.

    It should also be noted that this kind of model, while often used for face
    landmarking, is quite general and can be used for a variety of shape
    prediction tasks.  But here we demonstrate it only on a simple face
    landmarking task.
*/


#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
);
/*!
    ensures
        - returns an object D such that:    
            - D[i][j] == the distance, in pixels, between the eyes for the face represented
              by objects[i][j].
!*/

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{  
    try
    {
        dlib::array<array2d<unsigned char> > images_train, images_test;
        std::vector<std::vector<full_object_detection> > faces_train, faces_test;
        
        int landmarknum = 194;
        string basename = "/home/sooda/data/helen/";
        FILE* fin = fopen("/home/sooda/data/helen/annotation1.txt", "r");
        char filename[80];
        //while(fscanf(fin, "%s%*c", filename) != EOF) {
        for(int i = 0; i < 500; i++) {
            fscanf(fin, "%s%*c", filename);
            string fullname = basename + filename;
            array2d<unsigned char> img;
            load_image(img, fullname.c_str());
            images_train.push_back(img);
            int tempx, tempy, tempw, temph;
            fscanf(fin, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
            rectangle rect(tempx, tempy, tempw, temph);
            std::vector<point> parts;
            for(int j = 0; j < landmarknum; j++) { 
                int x, y;
                fscanf(fin, "%d %d%*c", &x, &y);
                point pt(x, y);
                parts.push_back(pt);
            }
            std::vector<full_object_detection> face;
            full_object_detection fobj(rect, parts);
            face.push_back(fobj);
            faces_train.push_back(face);
        }

        shape_predictor_trainer trainer;
        trainer.set_oversampling_amount(300);
        trainer.set_nu(0.05);
        trainer.set_tree_depth(2);

        trainer.be_verbose();

        shape_predictor sp = trainer.train(images_train, faces_train);

        serialize("sp.dat") << sp;
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

double interocular_distance (
    const full_object_detection& det
)
{
    dlib::vector<double,2> l, r;
    double cnt = 0;
    // Find the center of the left eye by averaging the points around 
    // the eye.
    for (unsigned long i = 36; i <= 41; ++i) 
    {
        l += det.part(i);
        ++cnt;
    }
    l /= cnt;

    // Find the center of the right eye by averaging the points around 
    // the eye.
    cnt = 0;
    for (unsigned long i = 42; i <= 47; ++i) 
    {
        r += det.part(i);
        ++cnt;
    }
    r /= cnt;

    // Now return the distance between the centers of the eyes
    return length(l-r);
}

std::vector<std::vector<double> > get_interocular_distances (
    const std::vector<std::vector<full_object_detection> >& objects
)
{
    std::vector<std::vector<double> > temp(objects.size());
    for (unsigned long i = 0; i < objects.size(); ++i)
    {
        for (unsigned long j = 0; j < objects[i].size(); ++j)
        {
            temp[i].push_back(interocular_distance(objects[i][j]));
        }
    }
    return temp;
}

// ----------------------------------------------------------------------------------------

