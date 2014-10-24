#include "compute.h"
#include "xianduan.h"
#include "mainwindow.h"

const double Compute::conflictParam  = 0.5;
const double Compute::consistentParam = 50;
const int Compute::MAX_POS = 1000;
//const int Compute::SEQ_LEN = 200;
const int Compute::OBJ_PER_FRAME = 600;
const int Compute::abnorml_thresh = 300;

Compute::Compute()
{
    strcpy(trackName, "track");
    strcpy(backgroundFile, "bg.jpg");
    strcpy(aviImgName, "meet/meet_%03d.jpg");
    strcpy(inputAvi, "test.avi");
    strcpy(outputAvi, "test_output.avi");
    SEQ_LEN = total_frame;
}


void Compute::clear() {
    imageVec.clear();
    objVector.clear();
    id_frame_vec.clear();
}

void Compute::setFileName(string filename) {
    aviFilename = filename;
}

string Compute::getFileName() {
    return aviFilename;
}


//对目标建立索引
void Compute::initObjIndex() {
    cout << "creating index .. " << endl;
    vector<blobObj>::iterator it = objVector.begin(), it_end = objVector.end();
    for( ; it != it_end; ++it) {
        blobObj temp = *it;
        int id = temp.id;
        int frameNo = temp.strartFrame;
        vector<Mat> objs;
        for(int i = 0; i < temp.total; i++) {
            CvBlob temp_blob = temp.blob_vec[i];
            int left_x = temp_blob.x - temp_blob.w/2;
            int left_y = temp_blob.y - temp_blob.h/2;
    //		cout << left_x << " " << left_y << " " << temp_blob.w/2 << " " << temp_blob.h/2<< endl;
            if(left_x + temp_blob.w >= width) {
                left_x = width - temp_blob.w;
            }
            if(left_y + temp_blob.h >= height) {
                left_y = height - temp_blob.h;
            }
            if(left_x < 0) {
                left_x = 0;
            }
            if(left_y < 0) {
                left_y = 0;
            }
            //assert(left_x >= 0 && left_y >= 0);
            //assert(left_x + temp_blob.w < width && left_y + temp_blob.h < height);
            Mat obj = Mat(imageVec[frameNo+i-1], Rect(left_x, left_y, temp_blob.w, temp_blob.h));
            objs.push_back(obj);
    //		cout << id << " " << frameNo + i << endl;
    //		cout << "push in index" << endl;
        }
        id_frame_vec.push_back(objs);
    }
}

void Compute::loadXml() {
    FileStorage fs("track_gt.xml", FileStorage::READ);

    FileNode infolist = fs["infolist"];
    FileNodeIterator it = infolist.begin(), it_end = infolist.end();

    vector<int> frameBeginVector;
    vector<int> objNumVector;


    for( ; it != it_end; it++) {
        FileNode info = *it;
        //id 开始 个数
        FileNodeIterator it_info = info.begin();
        int begin = *(++it_info);
        int num = *(++it_info);
        frameBeginVector.push_back(begin);
        objNumVector.push_back(num);
    }

    FileNode tracklist = fs["tracklist"];
    FileNodeIterator tl_it = tracklist.begin(), tl_end = tracklist.end();

    blobObj obj;
    int cnt = 1;
    for(; tl_it < tl_end; tl_it++) {
        FileNode track = *tl_it;
        obj.id = cnt;
        obj.strartFrame = frameBeginVector[cnt-1];
        obj.blob_vec.clear();
        obj.total = track.size();

        CvBlob blob;
        FileNodeIterator box_it = track.begin(), box_end = track.end();
        for(; box_it != box_end; box_it++) {
            FileNode box = *box_it;
            blob.ID = cnt;
            blob.x = box["x"];
            blob.y = box["y"];
            blob.h = box["h"];
            blob.w = box["w"];
            blob.x = blob.x + blob.w / 2;
            blob.y = blob.y + blob.h / 2;

            obj.blob_vec.push_back(blob);
        }
        cnt++;
        objVector.push_back(obj);
    }
}

void Compute::loadYml() {
    vector<int> frameBeginVector;
    vector<string> objNameVector;
    char trackFile[20];
    sprintf(trackFile, "%s.txt", trackName);
    FileStorage fs(trackFile, FileStorage::READ);
    CvFileStorage *fs2 = cvOpenFileStorage(trackFile, 0, CV_STORAGE_READ);
    FileNode n = fs[trackName];
    if(n.type() != FileNode::SEQ) {
        cerr << "is not a sequence! FAIL "<< endl;
        return ;
    }
    FileNodeIterator it = n.begin(), it_end = n.end();
    int cnt = 0;
    for(; it != it_end; ++it) {
        cnt++;
        int frameBegin = (int) (*it)["FrameBegin"];
        string objName = (string)(*it)["VideoObj"];
//        cout << frameBegin << endl;
//        cout << objName << endl;
        frameBeginVector.push_back(frameBegin);
        objNameVector.push_back(objName);
    }
    float cord[MAX_POS];
    float size[MAX_POS];
    //	for(vector<string>::iterator it = objNameVector.begin(); it != objNameVector.end(); ++it) {
    //		string objName = *it;
    blobObj obj;
    for(int i = 0; i < cnt; i++) {
        string objName = objNameVector[i];
        int id  = 0;
        //sscanf(objName.c_str(), "%*s_obj%d", &id);
        const char *c  = objName.c_str();
        const char *cid;
        char objTag[20];
        sprintf(objTag, "%s_obj", trackName);
        if(strncmp(c, objTag, strlen(objTag)) == 0) {
            cid = c + strlen(objTag);
        }
        id = atoi(cid);
    //    cout << id << endl;
        for(int j = 0; j < MAX_POS; j++) {
            cord[j] = -1;
            size[j] = -1;
        }
        obj.id = id;
        obj.strartFrame = frameBeginVector[i];
        obj.blob_vec.clear();
  //      cout << objName << endl;
        //CvFileNode *objNode = fs2[objName.c_str()];
        CvFileNode *objNode = cvGetFileNodeByName(fs2, NULL, objName.c_str());
        //CvFileNode *posNode = objNode["Pos"];
        CvFileNode *posNode = cvGetFileNodeByName(fs2, objNode, "Pos");
        cvReadRawData(fs2, posNode,(void*) cord, "ff");
        CvFileNode *sizeNode = cvGetFileNodeByName(fs2, objNode, "Size");
        cvReadRawData(fs2, sizeNode,(void*) size, "ff");
        CvBlob blob;
        int cnt_pos = 0;
        for(int j = 0; j < MAX_POS && cord[j] > 0; j++) {
            blob.ID = id;
            blob.w = size[j] * width;
            blob.h = size[j+1] * height;
            blob.x = cord[j] * width;
            blob.y = cord[j+1] * height;
            j++;
            cnt_pos++;
            obj.blob_vec.push_back(blob);
        }
        obj.total = cnt_pos;
        objVector.push_back(obj);
    }
}

void Compute::init() {
   // loadXml();
   // loadYml();
    loadItl("1.itl");
    initObjIndex();

}

vector<CvBlob> Compute::getBlobsByFrame( int frameNo) {
    vector<CvBlob> blobs;
    vector<blobObj>::iterator it = objVector.begin();
    vector<blobObj>::iterator it_end = objVector.end();
    for( ; it < it_end; it++) {
        blobObj temp = *it;
        if(temp.strartFrame <= frameNo && temp.strartFrame + temp.total > frameNo) {
            CvBlob blob = temp.blob_vec[frameNo-temp.strartFrame];
            blobs.push_back(blob);
        }
    }
    return blobs;
}

void Compute::print_seq(int *seq, int cnt) {
    for(int i = 0; i < cnt; i++) {
        cout << seq[i] << " " ;
    }
    cout << endl;
}

int Compute::random(int a, int b) {
    return rand() % (b-a) + a;
}

vector<CvBlob> Compute::getBlobsById (int id) {
    vector<CvBlob> blobs;
    vector<blobObj>::iterator it = objVector.begin();
    vector<blobObj>::iterator it_end = objVector.end();
    for( ; it < it_end; it++) {
        blobObj temp = *it;
        if(temp.id == id) {
            blobs = temp.blob_vec;
        }
    }
    return blobs;
}

int Compute::getStartFrame(int id) {
    vector<CvBlob> blobs;
    vector<blobObj>::iterator it = objVector.begin();
    vector<blobObj>::iterator it_end = objVector.end();
    for( ; it < it_end; it++) {
        blobObj temp = *it;
        if(temp.id == id) {
            return temp.strartFrame;
        }
    }
    return -1;
}

void Compute::swap_int(int& a, int &b) {
    int temp  = a;
    a = b;
    b = temp;
}

void Compute::changeState(int **pseq, int objTotal) {
    int changeNum = 3;
    int *seq = *pseq;
    for(int i = 0; i < changeNum; i++) {
        int index1 = random(0, objTotal-1);
        int index2 = random(0, objTotal-1);
        swap_int(seq[index1], seq[index2]);
    }
}

void Compute::assignSeq(int *lhs, int *rhs, int cnt) {
    for(int i = 0; i < cnt; i++) {
        lhs[i] = rhs[i];
    }
}

long long Compute::getLossEnergy(int *seq, int cnt) {
    long long ret = 0.0;
//	return ret;
    if(SEQ_LEN > 0 ) { //固定长度
        for(int i = 0; i < cnt; i++) {
//			cout << "in getLossEnergy " << i << endl;
            int total = objVector[i].total;
            if(seq[i] + total >= SEQ_LEN) {
                ret += (seq[i] + total - SEQ_LEN) * objVector[i].blob_vec[0].h * objVector[i].blob_vec[0].w;
            }
        }
    }
//	cout << "loss : " << ret << endl;
    return ret;
}

long long Compute::getConflictEnergy(int *seq, int cnt) {
    long long ret = 0;
    int *end = new int[cnt+1];
    for(int i = 0; i < cnt; i++) {
        end[i] = seq[i] + objVector[i].total;
    }
    for(int i = 0; i < cnt; i++) {
        for(int j = i + 1; j < cnt; j++) {
            if((seq[i]-seq[j] > 0 && end[i]-end[j] < 0) || (seq[i]-seq[j] < 0 && end[i]-end[j] > 0)) { //两个目标有交集
                int startIndex = seq[i] > seq[j] ? seq[i] : seq[j];
                int endIndex = end[i] > end[j] ? end[j] : end[i];
                for(int k = startIndex; k < endIndex; k++) {
                    //Rect ri = Rect(objVector[i].blob_vec[j].
                    CvBlob blob_i = objVector[i].blob_vec[k-seq[i]];
                    CvBlob blob_j = objVector[j].blob_vec[k-seq[j]];
                    //注意这里可能有边界问题
                    Rect ri = Rect(blob_i.x-blob_i.w/2, blob_i.y-blob_i.h/2, blob_i.w, blob_i.h);
                    Rect rj = Rect(blob_j.y-blob_j.w/2, blob_j.y-blob_j.h/2, blob_j.w, blob_j.h);
                    Rect r = ri & rj;   //&转到定义可以参看rect求交集的源码
                    if(r.width > 0 && r.height > 0) {
                        ret += r.area();
                    }
//					cout << "in getConflictEnergy " << i << " " << j << " " <<  r.area() << endl;
                }
            }
        }
    }
    delete [] end;
 //   cout << "confilt Loss " << conflictParam * ret << endl;
    return conflictParam * ret;
}


//获取最小距离
double Compute::getMinDistance(vector<CvBlob>::iterator it_a, vector<CvBlob>::iterator it_b, int cnt) {
    int distance = (it_a->x - it_b->x) * (it_a->x - it_b->x) + (it_a->y - it_b->y)*(it_a->y - it_b->y);
    it_a++;
    it_b++;
    for(int i = 1; i < cnt; i++) {
        int temp = (it_a->x - it_b->x) * (it_a->x - it_b->x) + (it_a->y - it_b->y)*(it_a->y - it_b->y);
        if(temp < distance) {
            distance = temp;
        }
    }
    return distance;
}


//计算一致性代价
long long Compute::getConsistencyEnergy(int *seq, int cnt) {
    long long ret = 0;
    int *end = new int[cnt+1];
    for(int i = 0; i < cnt; i++) {
        end[i] = seq[i] + objVector[i].total;
    }
    for(int i = 0; i < cnt; i++) { //计算时序性损失代价
        for(int j = i + 1; j < cnt; j++) {
            int start_i = objVector[i].strartFrame, start_j = objVector[j].strartFrame;
            int end_i = start_i + objVector[i].total, end_j = start_j + objVector[j].total;
            if( (start_i - start_j > 0 && end_i - end_j < 0) || (start_i - start_j < 0 && end_i - end_j > 0)) { //原视频有交集
                int start_orig = max(start_i, start_j);
                int end_orig = min(end_i, end_j);
                int delta_orig = end_orig - start_orig;
        //		CvBlob blob_i = objVector[i].blob_vec[start_orig-start_i];
        //		CvBlob blob_j = objVector[j].blob_vec[st
                vector<CvBlob>::iterator it_a = objVector[i].blob_vec.begin() + (start_orig - start_i);
        //		vector<CvBlob>::iterator it_a = &objVector[j].blob_vec[start_orig-start_i];
                vector<CvBlob>::iterator it_b = objVector[j].blob_vec.begin() + (start_orig - start_j);
                double distance = getMinDistance(it_a, it_b, delta_orig);  //原本的能量
                double energy = exp(-1 * distance / delta_orig);
                if((seq[i]-seq[j] > 0 && end[i]-end[j] < 0) || (seq[i]-seq[j] < 0 && end[i]-end[j] > 0)) { //摘要视频也有交集
                    int start_index = max(seq[i], seq[j]);
                    int end_index = min(end[i], end[j]);
                    int delta_sys = end_index - start_index;
                    vector<CvBlob>::iterator it_a_ = objVector[i].blob_vec.begin() + (start_index - seq[i]);
                    //		vector<CvBlob>::iterator it_a = &objVector[j].blob_vec[start_orig-start_i];
                    vector<CvBlob>::iterator it_b_ = objVector[j].blob_vec.begin() + (start_index - seq[j]);
                    double distance_ = getMinDistance(it_a_, it_b_, delta_sys);
                    double energy_ = exp(-1 * distance_ / delta_sys);
                    energy -= energy_;
                    if(energy < 0) {
                        energy = 0;
                    }
                  //  cout << energy << " " << energy_ << endl;
                }
                ret += energy;
            }
        }
    }
    delete [] end;
 //   cout << "consistentLoss: " << ret << endl;
    return ret;
}


long long Compute::getEnergy(int *seq, int cnt) {
    //return getConflictEnergy(seq, cnt) + getLossEnergy(seq, cnt) + getConsistencyEnergy(seq, cnt);
    return getConflictEnergy(seq, cnt) + getLossEnergy(seq, cnt);
}


//计算能量 每三个映射到一帧
long long Compute::calculate(int * seq, int objTotal) {
    //return 0.1;
    return getEnergy(seq, objTotal);
}


//目标与背景的合并非常普遍，写个函数包装一下
void Compute::paste(Mat &bg, Mat obj, Rect rect, char* msg /* = "" */) {
    Mat imageRoi = bg(rect);
    addWeighted(bg(rect), 0, obj, 1, 0., imageRoi);
    if(strlen(msg) > 0 ) {
        rectangle(bg, rect , Scalar(0, 255, 255));
        Scalar color(255, 0, 0);
        if(strcmp(msg, "abnormal") == 0) {
            color = Scalar(0, 0, 255);
        }
        putText(bg, msg, Point(rect.x, rect.y), 1, 1, color);
    }
}


void Compute::SA(int** pseq, int objTotal) {
    int *seq = *pseq;
    int *temp_seq = new int[objTotal+1];
    assignSeq(temp_seq, seq, objTotal);
    int t = 100;
    long long min_cost = calculate( seq, objTotal);
    int nochage = 0;
    int cnt = 0;
    while(1) {
        int bchange = 0;
        int L = 1000;
        while(L > 0) {
            //交换n个目标
            assignSeq(temp_seq, seq, objTotal);
            changeState(&temp_seq, objTotal);
            cnt++;
            L--;
            long long  cost = calculate(temp_seq, objTotal);
         //   cout << "min: " << min_cost << " current_cost: " << cost << endl;
            if(cost < min_cost) {
        //		cout << "min: " << min_cost << " current_cost: " << cost << endl;
                min_cost = cost;
                assignSeq(seq, temp_seq, objTotal);
            //	print_seq(seq, objTotal);
                cout << "accepted" << endl;
                bchange = 1;
            }
            else if(cost > min_cost){
        //		cout << "min: " << min_cost << " current_cost: " << cost << endl;
                double d = cost - min_cost;
                double e = exp(-d/t);
                double r = (rand() % 100) / 100.0;
                if(e > r ) {
                    min_cost = cost;
                    assignSeq(seq, temp_seq, objTotal);
            //		print_seq(seq, objTotal);
                    cout << "accepted" << endl;
                    bchange = 1;
                }
            }
            if(cnt % 100 == 0) {
                cout << "time: " << cnt << " cost: " << min_cost << endl;
            }
        }
        if(nochage > 10) {
            break;
        }
        t = t * 0.9;  //温度下降
        if(bchange == 0) {   //统计连续不变的次数
            nochage++;
        }
        else {
            nochage = 0;
        }
        cout << "L: " << L << " t: " << t << " times: " << nochage << endl;
    }
}


void Compute::combine(char* filename, Mat bg, int *seq, int cnt) {
    //根据seq来合并
    cout << "combining " << endl;
    int maxFrameNo = 0;
    VideoWriter writer;
    //若是有规定帧长，则帧长以此为准，否则，以最大能达到的为标准
    if(SEQ_LEN > 0) {
        maxFrameNo = SEQ_LEN;
    }
    else {
        for(int i = 0; i < cnt; i++) {
            int temp = seq[i] + objVector[i].total;
            if(temp > maxFrameNo) {
                maxFrameNo = temp;
            }
        }
    }
    vector<Mat> result(maxFrameNo);
    for(int i = 0; i < maxFrameNo; i++) {
        bg.copyTo(result[i]);
    }
    for(int i = 0; i < cnt; i++) {
        int startFrame = seq[i];
        int objNum = objVector[i].total;
        for(int j = 0; j < objNum; j++) {
            //超出固定长度，则忽略（造成能量损失）
            if(startFrame+j >= maxFrameNo) {
                break;
            }
//			cout << i << " " << j << "combining..." << endl;
            CvBlob temp = objVector[i].blob_vec[j];
            int left_x = temp.x - temp.w/2;
            int left_y = temp.y - temp.h/2;
            if(left_x + temp.w >= width) {
                left_x = width - temp.w;
            }
            if(left_y + temp.h >= height) {
                left_y = height - temp.h;
            }
            if(left_x < 0) {
                left_x = 0;
            }
            if(left_y < 0) {
                left_y = 0;
            }
            Rect rect = Rect(left_x, left_y, temp.w, temp.h);
            char msg[20];
            sprintf(msg, "%d: %d", i, j);
            paste(result[startFrame + j], id_frame_vec[i][j], rect, msg);
#ifdef DISPLAY
            imshow("result", result[startFrame+j]);
            waitKey(50);
#endif
        }
    }
    writer.open(filename, 1145656920, 5.0, Size(width, height));
    for(int i = 0; i < maxFrameNo; i++) {
#ifdef DISPLAY
        imshow("image", result[i]);
        waitKey(50);
#endif
        writer.write(result[i]);
    }
}

void Compute::combineWithTrace(char* filename, Mat bg, int *seq, int cnt) {
    //¸ù¾ÝseqÀ´ºÏ²¢
    cout << "combineWithTrace " << endl;
    VideoWriter writer;
    vector<Mat> result(SEQ_LEN);

    //´ø¹ì¼£µÄ±³¾°
    for(int frameno = 0; frameno < SEQ_LEN; frameno++) {
        for(int i = 0; i < cnt; i++) {
            int startFrame = seq[i];
            int objNum = objVector[i].total;
            if(startFrame+ objNum -1 > frameno && startFrame <= frameno+1) {
                CvBlob temp = objVector[i].blob_vec[frameno-startFrame+1];
                if(frameno-startFrame >= 0) {
                    CvBlob priBlob = objVector[i].blob_vec[frameno-startFrame];
                    Point pt1(temp.x, temp.y+temp.h/2);
                    Point pt2(priBlob.x, priBlob.y+priBlob.h/2);
                    line(bg,pt1,pt2, Scalar(0,0,255), 1.5);
                }
            }
        }
        bg.copyTo(result[frameno]);
    }

    for(int i = 0; i < cnt; i++) {
        int startFrame = seq[i];
        int objNum = objVector[i].total;
        for(int j = 0; j < objNum; j++) {
            //³¬³ö¹Ì¶¨³¤¶È£¬ÔòºöÂÔ£¨Ôì³ÉÄÜÁ¿ËðÊ§£©
            if(startFrame+j >= SEQ_LEN) {
                break;
            }
            CvBlob temp = objVector[i].blob_vec[j];
            int left_x = temp.x - temp.w/2;
            int left_y = temp.y - temp.h/2;
            if(left_x + temp.w >= width) {
                left_x = width - temp.w;
            }
            if(left_y + temp.h >= height) {
                left_y = height - temp.h;
            }
            if(left_x < 0) {
                left_x = 0;
            }
            if(left_y < 0) {
                left_y = 0;
            }
            Rect rect = Rect(left_x, left_y, temp.w, temp.h);
            char msg[20];
            sprintf(msg, "%d: %d", i+1, j);
            if(abnorm_flag[i]) {
                strcpy(msg, "abnormal");
            }
            paste(result[startFrame + j], id_frame_vec[i][j], rect, msg);
        }
    }

    writer.open(filename, 1145656920, 10.0, Size(width, height));
    for(int i = 0; i < SEQ_LEN; i++) {
        writer.write(result[i]);
    }
    imwrite("trace.jpg", bg);
}


void Compute::aviToImg(char *avi_filename, char* img_filename) {
    VideoCapture cap1(avi_filename);
    int cntt = 0;
    char filename1[20];
    for(;;) {
        Mat frame;
        cap1 >> frame;
        if(frame.data == 0) {
            break;
        }
        sprintf(filename1, img_filename,  cntt++);
        imwrite(filename1, frame);
    }
    cap1.release();
}


void Compute::InitParam() {
    VideoCapture cap(inputAvi);
    total_frame = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << total_frame << endl;
    width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    cout << width  << " " << height << endl;
    cap.release();
}


//将每一帧存到vector中，这样操作的时候会容易一些。但是，这同时也造成内存的大量消耗，在视频很大，目标稀疏的条件下，无法使用
void Compute::setImageContainer() {
/*	char  imageName[20];
    int cnt1 = 0;
    for(int i = 0; i < total_frame; i ++) {
        sprintf(imageName, aviImgName, cnt1++);
        imageVec.push_back(imread(imageName));
    } */
    imageVec.resize(total_frame);
    VideoCapture cap(inputAvi);
    int i = 0;
    while(1) {
        Mat frame;
        cap >> frame;
        if(frame.data == 0) {
            break;
        }
        //使用copyTo解决所有图片都同一张的问题，这样就无须将视频先转化为图像了，同时提高了读取速度
        frame.copyTo(imageVec[i++]);
    }
    cap.release();
#ifdef DISPLAY
    for(int j = 0; j < total_frame; j++) {
        imshow("image", imageVec[j]);
        if(waitKey(50) > 0) {
            break;
        }
    }
#endif
}


//合成示例代码，达到了效果
void Compute::muilti_object_combine_sample() {
    VideoWriter writer;
    VideoCapture vcap(inputAvi);
    int codec = static_cast<int>(vcap.get(CV_CAP_PROP_FOURCC));
    double frameRate = vcap.get(CV_CAP_PROP_FPS);
    writer.open(outputAvi, codec, frameRate, Size(width, height));
    //	writer.open("bike_output.avi", 1145656920, 30.0, frameSize);
    vector<Mat> result;
    int tt = 0;
    Mat bb = imread(backgroundFile);
    while(tt < id_frame_vec[0].capacity() || tt < id_frame_vec[1].capacity()) {
        Mat temp_bg = bb;
        if(tt < id_frame_vec[0].capacity()) {
            CvBlob temp = objVector[0].blob_vec[tt];
            int left_x = temp.x - temp.w/2;
            int left_y = temp.y - temp.h/2;
            Rect rect = Rect(left_x, left_y, temp.w, temp.h);
            paste(temp_bg, id_frame_vec[0][tt], rect);
            //	Mat logo = Mat(bb, Rect(left_x, left_y, temp.w, temp.h));
            //Mat imageRoi = temp_bg(Rect(left_x, left_y, temp.w, temp.h));
            //addWeighted(temp_bg(Rect(left_x, left_y, temp.w, temp.h)), 0, id_frame_vec[0][tt], 1, 0., imageRoi);
        }
        if(tt < id_frame_vec[1].capacity()) {
            CvBlob temp = objVector[1].blob_vec[tt];
            int left_x = temp.x - temp.w/2;
            int left_y = temp.y - temp.h/2;
            Rect rect = Rect(left_x, left_y, temp.w, temp.h);
            paste(temp_bg, id_frame_vec[1][tt], rect);
            //	Mat logo = Mat(bb, Rect(left_x, left_y, temp.w, temp.h));
            //Mat imageRoi = temp_bg(Rect(left_x, left_y, temp.w, temp.h));
            //addWeighted(temp_bg(Rect(left_x, left_y, temp.w, temp.h)), 0, id_frame_vec[1][tt], 1, 0., imageRoi);
        }
        //writer.write(temp_bg);
        writer << temp_bg;
        result.push_back(temp_bg);
        imshow("image", temp_bg);
        waitKey(50);
        tt++;
    }
}


//根据得到的坐标，在原视频中画出目标
void Compute::boundingBox() {
    VideoCapture cap(inputAvi);
    //在原图像框出来了
    for(int i = 0; i < 130; i++) {
        Mat test;
        cap >> test;
        if(test.data == 0) {
            break;
        }
        vector<CvBlob> blobs = getBlobsByFrame(i);
        cout << blobs.capacity();
        if(blobs.capacity() == 0)  {
            continue;
        }
        vector<CvBlob>::iterator it = blobs.begin(), it_end = blobs.end();
        for( ; it < it_end; it++) {
            CvBlob temp = *it;
            rectangle(test, Rect(temp.x - temp.w/2, temp.y - temp.h/2, temp.w, temp.h), Scalar(0, 255, 255));
            imshow("image", test);
            waitKey(30);
        }
        waitKey(100);
    }
}


void Compute::singleObjectCombineSample(int id) {
    //合成
    int stopLen = 1;
    Mat bg = imread(backgroundFile);
    vector<CvBlob> blobs_;
    blobs_ = getBlobsById(id);
    int startFrame = getStartFrame(id)-1;
    char filename[20];
    vector<CvBlob>::iterator it = blobs_.begin(), it_end = blobs_.end();
    int cnt = 0;
    for( ; it != it_end; it++) {
        cnt++;
        startFrame++;
        if(cnt % stopLen != 0) {
            continue;
        }
        CvBlob temp = *it;
        int left_x = temp.x - temp.w/2;
        int left_y = temp.y - temp.h/2;
        //Mat imageRoi = bg(Rect(left_x, left_y, temp.w, temp.h));
        Rect rect = Rect(left_x, left_y, temp.w, temp.h);
        sprintf(filename, aviImgName, startFrame);
        Mat current_image = imread(filename);
        Mat obj = Mat(current_image, rect);
        paste(bg, obj, rect);
    //	addWeighted(bg(Rect(left_x, left_y, temp.w, temp.h)), 0, logo, 1, 0., imageRoi);
        imshow("image", bg);
        waitKey(50);
    }
    imshow("image", bg);
}


//简单映射规则
void Compute::seqSimpleMap(int *seq, int cnt) {
    for(int i = 0; i < cnt; i++) {
        seq[i] = i / 3;     //每三个目标从同一帧开始，new startFrame
    }
    //随机化
    for(int i = 0; i < cnt; i++) {
        int randNum = rand() % cnt;
        swap_int(seq[randNum], seq[i]);
    }
}


//使用固定帧长度的规则进行映射,随机一个0到LEN的数字开搞
//使用这种规则可能造成最后一些超出固定长度，造成丢失
void Compute::seqDefLenMap(int *seq, int cnt) {
    for(int i = 0; i < cnt; i++) {
        int index = rand() % SEQ_LEN;
        seq[i] = index;
    }
}


void Compute::changeVideoFormat(char* lhs, char* rhs) {
    VideoCapture cap(rhs);
    VideoWriter writer;
    int width_ = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int height_ = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    writer.open(rhs, 1145656920, 10.0, Size(width_, height_));
    while(1) {
        Mat frame;
        cap >> frame;
        if(frame.data == 0) {
            break;
        }
        cout << "writing... " << endl;
        writer.write(frame);
    }
    writer.release();
    cap.release();
}


void Compute::previewVideo(char* filename, const char* jpgFormat) {
    VideoCapture cap(filename);
    int cnt = 0;
    while(1) {
        Mat frame;
        cap >> frame;
        if(frame.data == 0) {
            break;
        }
        if(cnt > 500) {
            break;
        }
        imshow("image", frame);
        char name[20];
        sprintf(name, jpgFormat, cnt++);
        imwrite(name, frame);
        waitKey(10);
    }
    cap.release();
}


void Compute::convertImgToAvi(const char* jpgFormat, char* dst) {
    char jpgFileName[20];
    int count = 0;
    sprintf(jpgFileName, jpgFormat, count++);
    Mat img = imread(jpgFileName,-1);
    Size frameSize = img.size();
    VideoWriter writer;
    writer.open(dst, 1145656920, 30.0, frameSize);
    namedWindow("image", 1);
    while(!img.empty()) {
        cout << count << endl;
        cout << jpgFormat << endl;
        imshow("image", img);
        writer.write(img);
        cout << "writing.." << jpgFileName <<  endl;
        sprintf(jpgFileName, jpgFormat, count++);
        img = imread(jpgFileName, -1);
        if(waitKey(50) > 0) {
            break;
        }
    }
}


void Compute::videoFormatToAvi(char* filename) {
    char* pos = strstr(filename, ".");
    char name[20];
    memset(name, 0, sizeof(name));
    strncpy(name, filename, pos-filename);
    cout << name << endl;
    char temp[20];
    char outname[20];
    //sprintf(temp, "%s\/\%03d.jpg", name);
    string s1(name);
    string s2 = s1 + "/%03d.jpg";
    sprintf(outname, "%s.avi", name);
    previewVideo(filename, s2.c_str());
    convertImgToAvi(s2.c_str(), outname);
}


void Compute::getBackground() {
    Mat img1, img2;
    img1 = imread("img1.jpg");
    img2 = imread("img2.jpg");
    Rect rect(680, 0, img1.cols-680-1, img1.rows-1);
    Mat obj = Mat(img2, rect);
    paste(img1, obj, rect);
    imwrite("bg.jpg", img1);
}


void Compute::loadItl(string filename) {

    FILE *pFile;
    pFile = fopen(filename.c_str(), "r");

    int num;
    fscanf(pFile, "%d", &num);
    int id_num = 1;

    for(int i = 0; i < num; i++) {
        int fr_start, fr_end, id;
        fscanf(pFile, "%d %d %d", &id, &fr_start, &fr_end);
        int blobnum = fr_end - fr_start + 1;

        float *px = new float[blobnum+1];
        float *py = new float[blobnum+1];
        float *pw = new float[blobnum+1];
        float *ph = new float[blobnum+1];

        for(int j = 1; j <= blobnum; j++) {
            fscanf(pFile, "%f", &px[j]);
        }
        for(int j = 1; j <= blobnum; j++) {
            fscanf(pFile, "%f", &py[j]);
        }
        for(int j = 1; j <= blobnum; j++) {
            fscanf(pFile, "%f", &pw[j]);
        }
        for(int j = 1; j <= blobnum; j++) {
            fscanf(pFile, "%f", &ph[j]);
        }
        //¶ÔÓÚ³¬¹ýÐèÒª³¤¶ÈµÄ½øÐÐ²ð·Ö
        int crash_time = blobnum / SEQ_LEN;
        int current_index = 1;
        for(int crash_id = 0; crash_id <= crash_time; crash_id++) {
            current_index = crash_id * SEQ_LEN + 1;
            if(current_index > blobnum) {
                break;
            }
            blobObj obj;
            obj.id = id_num;

            if(blobnum > abnorml_thresh) {
                abnorm_flag[id_num] = true;
            } else {
                abnorm_flag[id_num] = false;
            }

            obj.strartFrame = fr_start + current_index -1;
            obj.blob_vec.clear();
            int crash_num = 0;
            if(current_index + SEQ_LEN <= blobnum) {
                crash_num = SEQ_LEN;
            }
            else {
                crash_num = blobnum - current_index + 1;
            }
          //  cout << "blobnum: " << blobnum << " current_index: " << current_index  << " num:" << crash_num << endl;
            obj.total = crash_num;
            for(int j = current_index; j < current_index + crash_num; j++) {
                CvBlob blob;
                blob.ID = id_num;
                blob.x = px[j] + pw[j] / 2;
                blob.y = py[j] + ph[j] / 2;
                blob.w = pw[j];
                blob.h = ph[j];
                obj.blob_vec.push_back(blob);
            }
            objVector.push_back(obj);
            id_num++;
        }
        delete [] px;
        delete [] py;
        delete [] pw;
        delete [] ph;
    }
}

void Compute::new_arrange(int **pseq, int *map) {
    int objnum = objVector.size();
    int * cnt = new int[objnum + 1];
//	int cnt[50];
    memset(cnt, 0, sizeof(cnt));
    for(int i = 0; i < objnum; i++) {
        cnt[i] = objVector[map[i]].total;
    }
    int fr_length = SEQ_LEN;
    int fr_objnum = OBJ_PER_FRAME;

    test_xianduanshu(pseq, cnt, objnum, fr_length, fr_objnum, map);
    //将不合理的映射去除
    delete[] cnt;
}

void Compute::clusting(int** pmap) {
    int objnum = objVector.size();
    //¶ÔÓÚÆðµãµÄx£¬y½øÐÐ¾ÛÀà
//	int* start_x = new int(objnum+1);
//	int* start_y = new int(objnum+1);
    int *start_x = new int[objnum];
    int *start_y = new int[objnum];
//	int start_x[50];
//	int start_y[50];
    for(int i = 0; i < objnum; i++) {
        start_x[i] = objVector[i].blob_vec[0].x;
        start_y[i] = objVector[i].blob_vec[0].y;
    }

    const int MAX_CLUSTERS = 30;
    Scalar colorTab[] =
    {
        Scalar(0, 0, 255),
        Scalar(0,255,0),
        Scalar(255,100,100),
        Scalar(255,0,255),
        Scalar(0,255,255),
        Scalar(50, 0, 255),
        Scalar(50,255,0),
        Scalar(255,50,100),
        Scalar(255,50,255),
        Scalar(50,255,255),
        Scalar(100, 0, 255),
        Scalar(1000,255,0),
        Scalar(255,150,100),
        Scalar(255,100,255),
        Scalar(100,255,255),
        Scalar(0, 150, 255),
        Scalar(0,255,150),
        Scalar(255,100,150),
        Scalar(255,150,255),
        Scalar(150,255,255)
    };

    Mat img(height, width, CV_8UC3);
    //RNG rng(12345);
    int clusterCount = 5;
    int sampleCount = objnum;
    //Mat points(sampleCount, 1, CV_32FC2);
    Mat points(sampleCount, 1, CV_32FC2),labels;
//	Mat points = cvCreateMat(sampleCount, 1, CV_32FC2);
//	Mat labels;
    clusterCount = min(clusterCount, sampleCount);
    Mat centers(clusterCount, 1, points.type());

    //for(int k = 0; k < clusterCount; k++) {
    //	Point center;
    //	center.x = rng.uniform(0, width);
    //	center.y = rng.uniform(0, height);
    //}
    for(int i = 0; i < objnum; i++) {
        double x = start_x[i];
        double y = start_y[i];
        points.at<Point2f>(i) = Point(x,y);
    }

//	randShuffle(points, 1, &rng);

    kmeans(points, clusterCount, labels,
        TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);

     img = Scalar::all(0);

     int *map = *pmap;

     int map_cnt = 0;
     for(int i = 0; i < clusterCount; i++) {
         for(int j = 0; j < sampleCount; j++) {
             if(labels.at<int>(j) == i) {
                map[map_cnt++] = j;
             }
         }
     }

     for(int i = 0; i < sampleCount; i++ )
     {
         int clusterIdx = labels.at<int>(i);
         cout << clusterIdx <<" ";
         Point ipt = points.at<Point2f>(i);
         circle(img, ipt, 2, colorTab[clusterIdx], CV_FILLED, CV_AA );
        // char msg[5];
        // sprintf(msg, "%d", i);
        // putText(img, msg, ipt, 1, 1, Scalar(255, 0, 0));
     }
     cout << endl;

     //imshow("clusters", img);

    // waitKey();
     delete[] start_x;
     delete[] start_y;
}

void Compute::generateItl() {
    freopen("meet.itl", "w", stdout);
    loadXml();
    int tracknum = objVector.size();
    cout <<  tracknum << endl;
    for(int i = 0; i < tracknum; i++) {
        int start_fr = objVector[i].strartFrame;
        int end_fr = start_fr + objVector[i].total - 1;
        cout << i+1 << " " << start_fr << " " << end_fr << endl;
        int num = objVector[i].total;
        //x
        cout << objVector[i].blob_vec[0].x - objVector[i].blob_vec[0].w / 2;
        for(int j = 1; j < num; j++) {
            cout << " " << objVector[i].blob_vec[j].x - objVector[i].blob_vec[j].w / 2;
        }
        cout << endl;
        //y
        cout << objVector[i].blob_vec[0].y - objVector[i].blob_vec[0].h / 2;
        for(int j = 1; j < num; j++) {
            cout << " " << objVector[i].blob_vec[j].y - objVector[i].blob_vec[j].h / 2;
        }
        cout << endl;
        //w
        int w = 30;
        cout << objVector[i].blob_vec[0].w;
        //cout << w;
        for(int j = 1; j < num; j++) {
            cout << " " << objVector[i].blob_vec[j].w;
            //cout << " " << w;
        }
        cout << endl;
        //h
        int h = 30;
        //cout << h;
        cout << objVector[i].blob_vec[0].h;
        for(int j = 1; j < num; j++) {
            cout << " " << objVector[i].blob_vec[j].h;
        //	cout << " " << h;
        }
        cout << endl;

        int omega = 1;
        cout << omega;
        for(int j = 1; j < num; j++) {
            cout << " " << omega ;
        }
        cout << endl;
    }
}

void Compute::test_xianduanshu(int**pseq, int* cnt, int n, int w, int h, int*map)
{
    if(h > n) h = n;
    //max_length = w;
    XianDuanShu xds(w);
    xds.build(1, h, 1);
    int *start_fr = new int[h+1];
//	int start_fr[30];
//	memset(start_fr, 0, sizeof(start_fr));
    for(int i = 0; i < h; i++) {
        start_fr[i] = 1;
    }
    int *seq = *pseq;

    for(int i = 0; i < n ; i++) {
        if(xds.max_value[1] < cnt[i]) {
           // cout << "w: " << w << " xds: " << xds.max_value[1] << " cnt[i]: " << cnt[i] << endl;
            seq[map[i]] = -1;
        }
        else {
            int level = xds.query(cnt[i], 1, h, 1);
            seq[map[i]] = start_fr[level];
            start_fr[level] += cnt[i];
        }
    }
    delete[] start_fr;
}
