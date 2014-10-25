#include <iostream>
#include <stdio.h>
#include <map>
#include <string>
#include <string.h>

#define LANDMARK_NUM  194


using namespace std;

typedef struct Point {
    int x, y;
    Point() {
        x = 0; 
        y = 0;
    }
} Point2i;

class ObjDetect {
    public:
    ObjDetect() {
        x = -1;
        y = -1;
        w = -1;
        h = -1;
    }
    ObjDetect(const ObjDetect& obj) {
        x = obj.x;
        y = obj.y;
        w = obj.w;
        h = obj.h;
        for(int i = 0; i < LANDMARK_NUM; i++) {
            landmark[i].x = obj.landmark[i].x;
            landmark[i].y = obj.landmark[i].y;
        }

    }
    ObjDetect(int x_, int y_, int w_, int h_) { 
        x = x_;
        y = y_;
        w = w_;
        h = h_;
    }
    public:
        int x,  y, w, h;
        Point2i landmark[LANDMARK_NUM];
        
};

int main() {
    freopen("output.txt", "w", stdout);
    std::map<string, ObjDetect> annotation;
    char filename[80];
    FILE *finBB = fopen("helen_bb.txt", "r");
    while(fscanf(finBB, "%s%*c", filename) != EOF) {
        int tempx, tempy, tempw, temph;
        fscanf(finBB, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        if(tempx < 0) tempx = 0; 
        if(tempy < 0) tempy = 0; 
        if(tempw < 0) tempw = 0; 
        if(temph < 0) temph = 0; 
        ObjDetect bbox(tempx, tempy, tempw, temph); 
        annotation[filename] = bbox;
    }
    fclose(finBB);

    FILE* finBBtest = fopen("helen_test_bb.txt", "r");
    while(fscanf(finBBtest, "%s%*c", filename) != EOF) {
        int tempx, tempy, tempw, temph;
        fscanf(finBB, "%d %d %d %d%*c", &tempx, &tempy, &tempw, &temph);
        if(tempx < 0) tempx = 0; 
        if(tempy < 0) tempy = 0; 
        if(tempw < 0) tempw = 0; 
        if(temph < 0) temph = 0; 
        ObjDetect bbox(tempx, tempy, tempw, temph); 
        annotation[filename] = bbox;
    }
    FILE *finLandmark = fopen("1.txt", "r");
    while(fscanf(finLandmark, "%s%*c", filename) != EOF) {
        string fileString = string(filename) + ".jpg";
        ObjDetect bbox = annotation[fileString]; 
        for(int i = 0; i < LANDMARK_NUM; i++) { 
            float x, y;
            fscanf(finLandmark, "%f , %f%*c", &x, &y);
            if(x < 0) x = 0;
            if(y < 0) y = 0;
            bbox.landmark[i].x = x;
            bbox.landmark[i].y = y; 
        }
        annotation[fileString] = bbox;
    }
    std::map<string, ObjDetect>::iterator it = annotation.begin();
    for(; it != annotation.end(); it++){
        std::cout << it->first << std::endl;
        
        ObjDetect temp(it->second);
        std::cout << temp.x << " " << temp.y << " " << temp.w << " " << temp.h << std::endl;
        for(int i = 0; i < LANDMARK_NUM; i++) {
            std::cout << temp.landmark[i].x << " " << temp.landmark[i].y << std::endl; 
        }
    }
    return 0;

}
