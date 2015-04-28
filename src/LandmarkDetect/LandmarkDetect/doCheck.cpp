
#include "doCheck.h"


//////////////////////////////////////////////////////////////////////////////////////////////// 
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
bool  compute_coutor(Mat mat_, CvPoint &pt1, CvPoint &pt2);
static float GetMax(float a, float b);
static float GetMin(float a, float b);

cv::Mat HistogramEqualization(cv::Mat image, int thres1, int thres2,int channels)
{
	cv::Mat output; 
	float fvalue;
	image.copyTo(output);

 

	for(int i = 0; i < image.rows; i ++){
		for(int j = 0; j < channels * image.cols; j += channels){

			for(int k=0;k<channels;k++)
			{
				fvalue = image.at<uchar>(i, j+k);
				 
				fvalue = 255 * (fvalue-thres1) / (thres2 - thres1);

				output.at<uchar>(i, j+k) = GetMin(255, GetMax(0, fvalue)); 

				 
			}

		}
	}
	return output;
}




bool doCheck_eye(Mat image_in, std::vector<cv::Point2f>& point2f_vector,int upindex,int midindex,int downindex,int conindex1,int conindex2)
{
	int nPoint = point2f_vector.size();	
	if(upindex>=nPoint||midindex>=nPoint||downindex>=nPoint ||conindex1>=nPoint||conindex2>=nPoint) return FALSE;

	Point2f  pfup,pfmid,pfdown,pfTemp;
	int xmin=0,xmax=0,ymin=0,ymax=0,deltay1,deltay2; 

	pfup  = point2f_vector[upindex];
	xmin = xmax = pfup.x;
	ymin  =ymax = pfup.y; 	 
	pfmid  = point2f_vector[midindex];

	deltay1 = pfmid.y - ymax;
	if(xmin>pfmid.x) xmin = pfmid.x;
	if(xmax<pfmid.x) xmax = pfmid.x;
	if(ymin>pfmid.y) ymin = pfmid.y;
	if(ymax<pfmid.y) ymax = pfmid.y;		 
	 
	pfdown  = point2f_vector[downindex];
	deltay2 = pfdown.y - ymax;
	if(xmin>pfdown.x) xmin = pfdown.x;
	if(xmax<pfdown.x) xmax = pfdown.x;		                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
	if(ymin>pfdown.y) ymin = pfdown.y;
	if(ymax<pfdown.y) ymax = pfdown.y;

	ymin = ymin - (deltay2+ deltay1)/2;
	ymax = ymax + (deltay2+ deltay1)/2;
		

	if(xmax-xmin<abs(pfdown.y-pfup.y))
	{
		int delta = (pfdown.y-pfup.y)-(xmax-xmin);
		xmin = xmin -delta/2;
		xmax = xmax + delta/2;			
	}
	 
	pfTemp  = point2f_vector[conindex1];
	if(ymin<pfTemp.y) ymin = pfTemp.y + 5;
			 
	pfTemp  = point2f_vector[conindex2];
	if(ymin<pfTemp.y) ymin = pfTemp.y + 5;	

	float  total;
	int cols  = xmax-xmin+1;
	int rows = ymax-ymin+1;
	int *linesvalue = new int[rows];
	float point_mean=0;
	//cout<<"xmax:"<<xmax<<"xmin:"<<xmin<<"ymax:"<<ymax<<"ymin:"<<ymin<<endl;
	for(int i=ymin;i<=ymax;i++) //y
	{
		total = 0.0;
		for(int col=xmin;col<=xmax;col++) //x
		{
			
			total += image_in.at<uchar>(i,col);//用Vec3b也行		
			
		}
		linesvalue[i-ymin] = total/cols;
		point_mean +=  total;
	}
	
	point_mean = point_mean/((ymax-ymin+1)*(xmax-xmin+1));
	//cout<<"xmax:"<<xmax<<"xmin:"<<xmin<<"ymax:"<<ymax<<"ymin:"<<ymin<<"point_mean:"<<point_mean<<endl;
	point_mean *= 1.1;
	if(point_mean>250) point_mean = 250;
	
	
	Mat grayImg = HistogramEqualization(image_in, point_mean/3, point_mean,1);	 
	
	for(int i=ymin;i<=ymax;i++) //y
	{
		total = 0.0;
		for(int col=xmin;col<=xmax;col++) //x
		{			
			total += grayImg.at<uchar>(i,col);				
		}
		linesvalue[i-ymin] = total/cols;		 
	}

	int *linesvalue_pingjun = new int[rows];
	int num;
	float  total1;
	int linesvalue_pingjun_hight = 0;
	for(int i=0;i<rows;i++)
	{
		total = 0.0;
		
		num=0;
		for(int j=i-1;j<=i+ 1;j++)
		{
			if(j>=0&&j<rows)
			{
				num++;
				total += linesvalue[j];
			}
		}

		linesvalue_pingjun[i] = total/num;
		if(linesvalue_pingjun_hight<linesvalue_pingjun[i]) linesvalue_pingjun_hight = linesvalue_pingjun[i];
	}
	int *linesvalue_delta1 = new int[rows];
	int *linesvalue_delta2 = new int[rows];
	int delta;
	total1 = 0.0;
	int linesvalue_delta1_hight = 0;
	int linesvalue_delta2_hight = 0;
	for(int i=0;i<rows;i++)
	{
		num=0;
		total = 0.0;
		 
		for(int j=i+1;j<i+4;j++)
		{
			if(j>=0&&j<rows)
			{
				num++;
				delta = linesvalue_pingjun[j]-linesvalue_pingjun[i];
				if(abs(total)<abs(delta)) total = delta;				 
			}
		}
		if(num==0)  
		{
			for(int k=i-4;k<i-1;k++)
			{
				if(k>=0&&k<rows)
				{			
					delta = linesvalue_pingjun[k]-linesvalue_pingjun[i];
					if(abs(total)<abs(delta)) total = delta;				 
				}
			}
			 
		}
		 
		linesvalue_delta1[i] = total;
		total1 += abs(linesvalue_delta1[i]);
		if(abs(linesvalue_delta1_hight)<abs(linesvalue_delta1[i])) linesvalue_delta1_hight = abs(linesvalue_delta1[i]);
	}
	int  linesvalue_delta1_pingjun =  total1/rows;

	total1 = 0.0;
	for(int i=0;i<rows;i++)
	{
		num=0;
		total = 0.0;
		
		for(int j=i+1;j<i+5;j++)
		{
			if(j>=0&&j<rows)
			{
				num++;
				delta = linesvalue_delta1[j]-linesvalue_delta1[i];
				if(abs(total)<abs(delta)) total = abs(delta);	
				 
			}
		}
		if(num==0)  
		{
			for(int k=i-5;k<i-1;k++)
			{
				if(k>=0&&k<rows)
				{
					delta = linesvalue_delta1[k]-linesvalue_delta1[i];
					if(abs(total)<abs(delta)) total = abs(delta);	 
					 
				}
			}
			 
		}
		 
		linesvalue_delta2[i] = total;
		total1 += abs(linesvalue_delta2[i]);
		if(abs(linesvalue_delta2_hight)<abs(linesvalue_delta2[i])) linesvalue_delta2_hight = linesvalue_delta2[i];
	}

	int  linesvalue_delta2_pingjun =  total1/rows;		 

	int yup = -1;
	int ydown = -1; 
	int col1 = -1,col2=-2;
	for(int i=0;i<rows*3/5;i++)
	{
		if(linesvalue_delta1[i]<0&&linesvalue_delta1[i+1]<0)
		if((linesvalue_pingjun[i]>linesvalue_pingjun_hight*2/3&&abs(linesvalue_delta1[i])>linesvalue_delta1_pingjun&&abs(linesvalue_delta1[i])>linesvalue_delta1_hight/4) ||
			(linesvalue_pingjun[i]>linesvalue_pingjun_hight*2/3&&linesvalue_delta2[i]>linesvalue_delta2_pingjun&& linesvalue_delta2[i]>linesvalue_delta2_hight/4))
		{			 
			col1 = i;			
			break;
		}
	}
/*	if(-1!=col1)
	{
		int  col = col1;
		int  cvalue = linesvalue_pingjun[col];
		for(int i=col1-1;i<col1-6;i--)
		{
		    if(i>=0&&linesvalue_pingjun[i]>linesvalue_pingjun[col])
			{
				col = i;
				cvalue = linesvalue_pingjun[col]; 
			}
			else if(linesvalue_pingjun[i]<cvalue) break;
		}
		col1 = col;
	}*/

	 
	for(int i=rows-1;i>=rows*2/5;i--)
	{
		if(linesvalue_delta1[i]>0&&linesvalue_delta1[i-1]>0)
		if((linesvalue_pingjun[i]>linesvalue_pingjun_hight*3/5&&abs(linesvalue_delta1[i])>linesvalue_delta1_pingjun*2/3 && abs(linesvalue_delta1[i])>linesvalue_delta1_hight/5)||
			(linesvalue_pingjun[i]>linesvalue_pingjun_hight*3/5&&abs(linesvalue_delta2[i])>abs(linesvalue_delta2_pingjun)&& abs(linesvalue_delta2[i])>abs(linesvalue_delta2_hight)/4))
		{			 
			col2 = i;
			break;

		}
	}

/*	if(-1!=col2)
	{
		int  col = col2;
		int  cvalue = linesvalue_pingjun[col];
		for(int i=col2+1;i<col2+6;i++)
		{
		    if(i<rows&&linesvalue_pingjun[i]>linesvalue_pingjun[col])
			{
				col = i;
				cvalue = linesvalue_pingjun[col]; 
			}
			else if(linesvalue_pingjun[i]<cvalue) break;
		}
		col2 = col;
	}*/
	
	yup = col1;
	ydown = col2; 	 
	//cout<<"dddddddddddddddddddddddd yup:"<<yup<<"ydown:"<<ydown<<endl;
	if(-1!=yup)	point2f_vector[upindex].y = ymin+yup;  
	if(-1!=ydown) point2f_vector[downindex].y = ymin+ydown;
	return TRUE;
}

  
bool roi_mouth(Mat image_in, std::vector<cv::Point2f>& point2f_vector,int leftindex,int rightindex,int upindex,int downindex,cv::Point2f &roi_left,cv::Mat &rect_result)
{
	bool bflag = false; 
	
	
	IplImage Ipllmage_in = image_in;  

	do
	{
		float xmin = point2f_vector[leftindex].x;
		float xmax = point2f_vector[rightindex].x;
		float ymin = point2f_vector[upindex].y;
		float ymax = point2f_vector[downindex].y;
		float deltax = xmax - xmin;
		float deltay = ymax - ymin;		 

		float iwidth,iheight;
		iwidth =  image_in.cols;
		iheight = image_in.rows;

		if(iwidth<0||iheight<0) break;
		if(xmin<0||xmin>iwidth) break;
		if(xmax<0||xmax>iwidth) break;
		if(ymin<0||ymin>iheight) break;
		if(ymax<0||ymax>iheight) break;

		int rect_x= xmin-deltax/4;
		int rect_y=  ymin-deltay*2/5;
		int rect_w= deltax*3/2;
		int rect_h= deltay*9/5;


		if(rect_x<0) rect_x=0;	 
		if(rect_x+rect_w>=iwidth )
		{
			rect_x=0;
			rect_w = iwidth;
		}

		if(rect_y+rect_h>=iheight )
		{
			rect_y=iheight-rect_h-5;
			if(rect_y<0) 
			{
				rect_y=0;
				rect_h = iheight;
			}
		}

		try
		{

			CvRect roi_rect = cvRect(rect_x, rect_y, rect_w, rect_h);	
			cvSetImageROI(&Ipllmage_in, roi_rect);
			CvSize roi_size = cvGetSize(&Ipllmage_in);
			IplImage *img_mouth = cvCreateImage(roi_size,
                               Ipllmage_in.depth,
                               Ipllmage_in.nChannels);
			cvCopy(&Ipllmage_in, img_mouth, NULL);
			cvResetImageROI(&Ipllmage_in);
			roi_left.x = xmin-deltax/4;
			roi_left.y = ymin-deltay*2/5;

			rect_result =	cv::Mat(img_mouth,1);
			cvReleaseImage(&img_mouth);
		}
		catch(...)
		{

		}

		bflag = true; 
	}while(0);

	return bflag;
}

bool  doCheck_mouth(Mat image_in, std::vector<cv::Point2f>& point2f_vector,int leftindex,int rightindex,int upindex,int downindex)
{
	//cout<<"doCheck_mouth 1"<<endl;
	bool bflag = false;

	int point2f_vector_size = point2f_vector.size();

	if(point2f_vector_size==0) return false;	
	do
	{
		if(point2f_vector_size==0) break;
		if(point2f_vector_size<=leftindex) break;
		if(point2f_vector_size<=rightindex) break;
		if(point2f_vector_size<=upindex) break;
		if(point2f_vector_size<=downindex) break;
			
			

		if(point2f_vector[leftindex].x>=point2f_vector[rightindex].x||point2f_vector[leftindex].x>=point2f_vector[upindex].x||point2f_vector[leftindex].x>=point2f_vector[downindex].x) break;
		if(point2f_vector[rightindex].x<=point2f_vector[upindex].x||point2f_vector[rightindex].x<=point2f_vector[downindex].x) break;


		cv::Point2f  roi_left(0,0);
		cv::Mat  matMouth;
		 


		try
		{
			bflag	= roi_mouth(image_in,   point2f_vector, leftindex, rightindex, upindex, downindex,roi_left,matMouth);
		}
		catch(...)
		{
		}

		if(false==bflag) break;
		

		int rows = matMouth.rows, cols = matMouth.cols;

		if(rows<=0 || cols<=0) break;

		cv::Mat  float_roi(rows,cols,CV_32FC1,Scalar(1));
		cv::Mat  bi_roi(rows,cols,CV_8UC1,Scalar(1));	 
	 
		float r,g,b,bb,rr,result,fmin = 10000,fmax=-10000;
 		for(int i=0;i<rows;i++)
		{
			for(int j=0;j<cols;j++)
			{
				 b = matMouth.at<uchar>(i,3*j);
				 g = matMouth.at<uchar>(i,3*j+1);
				 r = matMouth.at<uchar>(i,3*j+2);
			 
				 bb = pow(b,(float)0.391);
				 rr = pow(r,(float)0.609);
				 if(bb*rr<0.0001||g<0.001) result = 0;
				 else	 result = log(g/(bb*rr));			
		 
				 float_roi.at<float>(i,j) = result;
				 if(fmin>result) fmin = result;
				 if(fmax<result) fmax = result;
			 		 
			}
		}	 
		//AfxMessageBox("doCheck_mouth 2");
		//cout<<"doCheck_mouth 2"<<endl;
		float min_c,delta;

	
		float otsu_ = otsu(float_roi, fmin, fmax);
		//AfxMessageBox("doCheck_mouth 3");
		//cout<<"doCheck_mouth 3"<<endl;
		for(int i=0;i<rows;i++)
		{
	
			for(int j=0;j<cols;j++)
			{	 
				result = float_roi.at<float>(i,j);


				if(result<otsu_)
					bi_roi.at<uchar>(i,j) =   255; 
				else bi_roi.at<uchar>(i,j) =   0;		 
			}
		}

	//	cv::imshow("imageaaa", bi_roi);
		
		CvPoint pt1,pt2; 
		pt1.x=0;
		pt1.y=0;
		pt2.x=0;
		pt2.y=0;

		bool bflag=false;
		try
		{
			bflag = compute_coutor(bi_roi, pt1, pt2);
		}catch(...)
		{

		}

		if(false==bflag) break;
		//AfxMessageBox("doCheck_mouth 5");
		//cout<<"doCheck_mouth 5"<<endl;

		matMouth.release();
		float_roi.release();
		bi_roi.release();


		cv::Point2f mouth_left;
		cv::Point2f mouth_right;

		mouth_left.x = roi_left.x + pt1.x;
		mouth_left.y = roi_left.y +pt1.y;
		mouth_right.x = roi_left.x +pt2.x;
		mouth_right.y = roi_left.y +pt2.y;

// 		CString strTest;
// 		strTest.Format("left:%f-%f right%f-%f left:%f-%f right:%f-%f",point2f_vector[leftindex].x,point2f_vector[leftindex].y,point2f_vector[rightindex].x,point2f_vector[rightindex].y,mouth_left.x,mouth_left.y,mouth_right.x,mouth_right.y);
// 		//AfxMessageBox(strTest);

		if(mouth_left.x>=mouth_right.x||mouth_left.x>=point2f_vector[upindex].x||mouth_left.x>=point2f_vector[downindex].x) break;		
		if(mouth_right.y<point2f_vector[upindex].y||mouth_right.y>point2f_vector[downindex].y) break;

		int mouth_center_x = (point2f_vector[upindex].x+point2f_vector[downindex].x)/2;
		int delta_x1 = mouth_center_x-mouth_left.x;
		int delta_x2 = mouth_right.x -mouth_center_x;
		float f_delta = (float)delta_x1/((float)delta_x2);

		if(f_delta>1.45||f_delta<0.73) break;

		int corner_miny;
		int corner_maxy;
		if(mouth_left.y>mouth_right.y)
		{
			corner_miny = mouth_right.y;
			corner_maxy = mouth_left.y;
		}
		else
		{
			corner_miny = mouth_left.y;
			corner_maxy = mouth_right.y;
		}
		int delta_y = point2f_vector[downindex].y-point2f_vector[upindex].y;

		if(corner_maxy-point2f_vector[downindex].y>delta_y/3) break;
		if((point2f_vector[upindex].y-corner_miny) >delta_y/4.0&&(corner_maxy-corner_miny)>delta_y/4.0) break;




	//	int delta_x1 = mouth_center_x-mouth_left.x;
	//	int delta_x2 = mouth_right.x -mouth_center_x;
	//	float f_delta = (float)delta_x1/((float)delta_x2);

		
		//cout<<"doCheck_mouth 6"<<point2f_vector[leftindex].x<<point2f_vector[leftindex].y<<point2f_vector[rightindex].x<<point2f_vector[rightindex].y<<endl;
		//cout<<"doCheck_mouth 7"<<mouth_left.x<<mouth_left.y<<mouth_right.x<<mouth_right.y<<endl;

		point2f_vector[leftindex].x  = mouth_left.x;
		point2f_vector[leftindex].y  = mouth_left.y;
		point2f_vector[rightindex].x = mouth_right.x;
		point2f_vector[rightindex].y = mouth_right.y;


		//cout<<"doCheck_mouth 8"<<endl;
		

		bflag = true;
	}while(0);

	return bflag;
}


float GetMax(float a, float b)
{
	return a>b ? a: b;
}

float GetMin(float a, float b)
{
	return a<b ? a: b;
}

std::vector<cv::Point2f> _ReadPTS(const char *filename)
{
	std::vector<cv::Point2f>   point2fs;
	FILE*   stream = NULL;
	char    rbuffer[256];
	char    *ptemp1,*ptemp2;
	long    nTotalPoint;
	cv::Point2f pointfvalue;
	do
	{
		if(NULL==filename) break;
		stream = fopen(filename,"r"); 
		if(NULL==stream)   break;
		 
		do
		{
			ptemp1 = fgets(rbuffer,256,stream);
			ptemp2 = strstr(rbuffer,"n_points:");
		}while ((NULL!=ptemp1)&&(NULL==ptemp2));

		if(NULL==ptemp2) break;
		ptemp1 = ptemp2 + strlen("n_points:");
		nTotalPoint = strtol(ptemp1,NULL,10);
		if(0>=nTotalPoint) break;

		ptemp1 = fgets(rbuffer,256,stream);
		for(int i=0;i<nTotalPoint;i++)
		{
			ptemp1 = fgets(rbuffer,256,stream);
			pointfvalue.x = strtod(ptemp1,&ptemp2);
			pointfvalue.y = strtod(ptemp2,NULL);
			point2fs.push_back(pointfvalue);
		}

	}while(0);

	if( stream)  
		fclose( stream );
    
	return point2fs;
}

void  _WritePTS(const char *filename,std::vector<cv::Point2f> totalpoint2f)
{
	FILE*   stream = NULL;
	char    rbuffer[256];
	int point_num =totalpoint2f.size();

	do
	{
		if(0==point_num) break;
		if(NULL==filename) break;
		stream = fopen(filename,"w+"); 
		if(NULL==stream)   break;
		fprintf(stream,"version: 1\n");
		fprintf(stream,"n_points: %d\n",point_num);
		fprintf(stream,"{\n"); 
		for(int i=0;i<point_num;i++)
		{
			fprintf(stream,"%.3f %.3f\n",totalpoint2f[i].x,totalpoint2f[i].y);
		}
		fprintf(stream,"}\n"); 

	}while(0);

	if( stream)  
		fclose( stream );
}

void  _WriteTXT(const char *filename,std::vector<cv::Point2f> totalpoint2f)
{
	FILE*   stream = NULL;
	char    rbuffer[256];
	int point_num =totalpoint2f.size();

	do
	{
		if(0==point_num) break;
		if(NULL==filename) break;
		stream = fopen(filename,"w+"); 
		if(NULL==stream)   break;
		 
		fprintf(stream,"asm ");
		fprintf(stream,"%d ",point_num);		 
		 
		for(int i=0;i<point_num;i++)
		{
			fprintf(stream,"%.3f %.3f ",totalpoint2f[i].x,totalpoint2f[i].y);
		}
		 

	}while(0);

	if( stream)  
		fclose( stream );

}

float otsu(Mat A,float fmin,float fmax)
{
	int wid =  A.cols;
	int hei =  A.rows;

	float fT=0;


	int* h=new int[256];
	double* p=new double[256];
	double* u=new double[256];
	double* w=new double[256];

	

	do
	{

		if(fmax==fmin) 
		{
			fmax=1000.0;
			fmin=-1000.0;
		}
		if(wid<=0||hei<=0) break;

		Mat result(hei ,wid,CV_8UC1,Scalar(0));
		long N = wid * hei;

	
		for(int i = 0; i < 256; i++)
		{

			h[i] = 0;
			p[i] = 0;
			u[i] = 0;
			w[i] = 0;
		}

		for(int i = 0; i < hei; i++)
		{
			for(int j = 0; j < wid; j++)
			{
			
				result.at<uchar>(i,j) = (A.at<float>(i,j)-fmin)*255/(fmax-fmin) ;
			}

		}

		for(int i = 0; i < hei; i++)
		{
			for(int j = 0; j < wid; j++)
			{
				for(int k = 0; k < 256; k++)
				{
					if(result.at<uchar>(i,j) == k)
					{
						h[k]++;
					}
				}
			}
		}

		for(int i = 0; i < 256; i++)
			p[i] = h[i] / double(N);

		int T = 0;
		double uT,thegma2fang;
		double thegma2fang_max = -10000;

		for(int k = 0; k < 256; k++)
		{
			

			uT = 0;
			for(int i = 0; i <= k; i++)
			{
				u[k] += i*p[i];
				w[k] += p[i];
			}

			for(int i = 0; i < 256; i++)
				uT += i*p[i];

			if(0==w[k]*(1-w[k])) 
			{			
				//AfxMessageBox("dododododododododo");
				break;
			}

			thegma2fang = (uT*w[k] - u[k])*(uT*w[k] - u[k]) / (w[k]*(1-w[k]));

			if(thegma2fang > thegma2fang_max)
			{
				thegma2fang_max = thegma2fang;
				T = k;
			}
		}

//	float fT = (float)T*5.0/8.0;
		fT = (float)T;
		fT = fmin + fT*(fmax-fmin)/255.0;
	
	}while(0);

	delete []h;
	delete []p;
	delete []u;
	delete []w;


	return fT;	 
}

bool  compute_coutor(Mat mat_, CvPoint &pt1, CvPoint &pt2)
{
	IplImage  iplImg = mat_; 
    IplImage* pContourImg = NULL;  
    CvSeq * contour = NULL; 
    CvSeq *contmax = NULL;
	CvMemStorage * storage=NULL;
	bool bflag = false;

	//AfxMessageBox("compute_coutor 1");
	do
	{
		storage = cvCreateMemStorage(0); 

		if(NULL==storage) break;
		//AfxMessageBox("compute_coutor 2");
		int mode = CV_RETR_EXTERNAL;  
 
		cvFindContours( &iplImg, storage, &contour, sizeof(CvContour), mode, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));

		if(NULL==contour) break;
		//AfxMessageBox("compute_coutor 3");
		int area,maxArea = 10;//设面积最大值大于10Pixel 
		for(;contour;contour = contour->h_next) 
		{ 
			area = fabs(cvContourArea( contour, CV_WHOLE_SEQ )); //获取当前轮廓面积  


			if(area > maxArea)
			{
				contmax = contour;
				maxArea = area;
			}
		}
		if(NULL==contmax) break;
		//AfxMessageBox("compute_coutor 4");

		CvSeqReader reader;
		if(contmax && contmax->total )
		{
			float xmin,xmax,ymin,ymax; 

			cvStartReadSeq( contmax, &reader, 0 );

			CvPoint pt;            
			CV_READ_SEQ_ELEM( pt, reader );
			xmin = xmax = pt.x;
			ymin = ymax = pt.y;

			for(int  i = 1; i < contmax->total; i++ )
			{
				CV_READ_SEQ_ELEM( pt, reader );
				if( xmin > pt.x )
				{
					xmin = pt.x;
					pt1 = pt;
				}

				if( xmax < pt.x )
				{
					xmax = pt.x;
					pt2 = pt;
				}     
			}
		}

		//AfxMessageBox("compute_coutor 5");
	
		bflag = true;
	}while(0);

	if(contour)
		cvClearSeq(contour);
	if(contmax)
		cvClearSeq(contmax);
	if(storage)
		cvReleaseMemStorage(&storage);


	return bflag;
}