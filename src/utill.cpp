#include "util.h"
namespace customCV {

/**  @function Erosion  */
cv::Mat Erosion(cv::Mat src, int size, int type)
{
  cv::Mat dst;
  int erosion_type;
  int erosion_size = size;
  if( type == 0 ){ erosion_type = cv::MORPH_RECT; }
  else if( type == 1 ){ erosion_type = cv::MORPH_CROSS; }
  else if( type == 2) { erosion_type = cv::MORPH_ELLIPSE; }

  cv::Mat element = cv::getStructuringElement( erosion_type,
                                       cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       cv::Point( erosion_size, erosion_size ) );
  erode( src, dst, element );
  return dst;
}

cv::Mat  Dilation(cv::Mat src, int size, int type)
{
  cv::Mat dst;
  int dilation_type;
  int dilation_size = size;  
  if( type == 0 ){ dilation_type = cv::MORPH_RECT; }
  else if( type == 1 ){ dilation_type = cv::MORPH_CROSS; }
  else if( type == 2) { dilation_type = cv::MORPH_ELLIPSE; }

  cv::Mat element = cv::getStructuringElement( dilation_type,
                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       cv::Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( src, dst, element );
  return dst;
}

IplImage* Transform(IplImage* A,CvScalar avg_src, CvScalar avg_dst,CvScalar std_src, CvScalar std_dst, IplImage* mask = NULL)  
{  
	for(int i=0;i<3;i++)  
	{  
		for(int x=0;x<A->height;x++)  
		{  
			uchar *ptr=(uchar*)(A->imageData+x*A->widthStep);
			uchar * p_mask;
			if(mask != NULL) 
				p_mask = (uchar*)(mask->imageData+x*mask->widthStep);
			for(int y=0;y<A->width;y++)  
			{  
				if(mask != NULL) 
					if((int)p_mask[y] < 5)
						continue;	
				double tmp=ptr[3*y+i];  
				int t=(int)((tmp-avg_dst.val[i])*(std_src.val[i]/std_dst.val[i])+avg_src.val[i]);  
				t = t<0?0:t;  
				t = t>255?255:t;  
				ptr[3*y+i]=t;  
			}  
		}  
	}  
	return A;  
} 

//处理uchar类型
void  transformMat(cv::Mat& src, cv::Scalar avg_src, cv::Scalar avg_dst, cv::Scalar std_src, cv::Scalar std_dst, cv::InputArray maskMat) {
	cv::Mat mask = maskMat.getMat();
//	CV_Assert( mask.empty() || mask.type() == CV_8U );
	int rows = src.rows;
	int cols = src.cols;
	for(int i = 0; i < 3; i++) {
		for(int x = 0; x < rows; x++) {
			uchar *ptr = (uchar*)(src.data + x * src.step);
			uchar *pmask = NULL; 
			if(!mask.empty()) {
				pmask = (uchar*)(mask.data + x * mask.step);
			}
			for(int y = 0; y < cols; y++) {
				if(!mask.empty())
					if(pmask[y] < 5)
						continue;
				double tmp = ptr[3 * y + i];
				int t = (int)((tmp - avg_dst.val[i]) * (std_src.val[i]/std_dst.val[i]) + avg_src.val[i]);
				t = t < 0 ? 0 : t;
				t = t > 255 ? 255 : t;
				ptr[3*y + i] = t;
			}
		
		}
	}
}


void cvShowManyImages(char* title, int nArgs, ...)   
{  
  
    // img - Used for getting the arguments   
    IplImage *img;  
  
    // DispImage - the image in which input images are to be copied  
    IplImage *DispImage;  
  
    int size;  
    int i;  
    int m, n;  
    int x, y;  
  
    // w - Maximum number of images in a row   
    // h - Maximum number of images in a column   
    int w, h;  
  
    // scale - How much we have to resize the image  
    float scale;  
    int max;  
  
    // If the number of arguments is lesser than 0 or greater than 12  
    // return without displaying   
    if(nArgs <= 0) {  
        printf("Number of arguments too small....\n");  
        return;  
    }  
    else if(nArgs > 12) {  
        printf("Number of arguments too large....\n");  
        return;  
    }  
    // Determine the size of the image, and the number of rows/cols  from number of arguments   
    else if (nArgs == 1) {  
        w = h = 1;  
        size = 300;  
    }  
    else if (nArgs == 2) {  
        w = 2; h = 1;  
        size = 300;  
    }  
    else if (nArgs == 3 || nArgs == 4) {  
        w = 2; h = 2;  
        size = 300;  
    }  
    else if (nArgs == 5 || nArgs == 6) {  
        w = 3; h = 2;  
        size = 200;  
    }  
    else if (nArgs == 7 || nArgs == 8) {  
        w = 4; h = 2;  
        size = 200;  
    }  
    else {  
        w = 4; h = 3;  
        size = 150;  
    }  
  
    // Create a new 3 channel image0  
    DispImage = cvCreateImage( cvSize( 100+ size*w, 60 + size*h), 8, 3 );  
  
    // Used to get the arguments passed  
    va_list args;  
    va_start(args, nArgs);  
  
    // Loop for nArgs number of arguments  
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {  
  
        // Get the Pointer to the IplImage  
        img = va_arg(args, IplImage*);  
  
        // Check whether it is NULL or not  
        // If it is NULL, release the image, and return  
        if(img == 0) {  
            printf("Invalid arguments");  
            cvReleaseImage(&DispImage);  
            return;  
        }  
  
        // Find the width and height of the image  
        x = img->width;  
        y = img->height;  
  
        // Find whether height or width is greater in order to resize the image  
        max = (x > y)? x: y;  
  
        // Find the scaling factor to resize the image  
        scale = (float) ( (float) max / size );  
  
        // Used to Align the images  
        if( i % w == 0 && m!= 20) {  
            m = 20;  
            n+= 0 + size;  
        }  
  
        // Set the image ROI to display the current image  
        //cvSetImageROI(DispImage, cvRect(m, n, (int)( x/scale ), (int)( y/scale )));  
        cvSetImageROI(DispImage, cvRect(m, n, (int)( x/scale ), (int)( y/scale )));  
        //      cout<<"x="<<m<<"y="<<n<<endl;  
  
        // Resize the input image and copy the it to the Single Big Image  
        cvResize(img, DispImage);  
  
        // Reset the ROI in order to display the next image  
        cvResetImageROI(DispImage);  
    }  
  
    // Create a new window, and show the Single Big Image  
    //cvNamedWindow( title, 1 );  
    cvShowImage( title, DispImage);  
  
    /*cvWaitKey(0);*/  
    //cvDestroyWindow(title);  
  
    // End the number of arguments  
    va_end(args);  
  
    // Release the Image Memory  
    cvReleaseImage(&DispImage);  
} 


void LBP (IplImage *src,IplImage *dst)  
{  
    int tmp[8]={0};  
    CvScalar s;  
  
    IplImage * temp = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U,1);  
    uchar *data=(uchar*)src->imageData;  
    int step=src->widthStep;  
   
  
    for (int i=1;i<src->height-1;i++)  
      for(int j=1;j<src->width-1;j++)  
      {  
          int sum=0;  
          if(data[(i-1)*step+j-1]>data[i*step+j])  
            tmp[0]=1;  
          else  
            tmp[0]=0;  
          if(data[i*step+(j-1)]>data[i*step+j])  
            tmp[1]=1;  
          else  
            tmp[1]=0;  
          if(data[(i+1)*step+(j-1)]>data[i*step+j])  
            tmp[2]=1;  
          else  
            tmp[2]=0;  
          if (data[(i+1)*step+j]>data[i*step+j])  
            tmp[3]=1;  
      else  
            tmp[3]=0;  
          if (data[(i+1)*step+(j+1)]>data[i*step+j])  
            tmp[4]=1;  
          else  
            tmp[4]=0;  
          if(data[i*step+(j+1)]>data[i*step+j])  
            tmp[5]=1;  
          else  
            tmp[5]=0;  
          if(data[(i-1)*step+(j+1)]>data[i*step+j])  
            tmp[6]=1;  
          else  
            tmp[6]=0;  
          if(data[(i-1)*step+j]>data[i*step+j])  
            tmp[7]=1;  
          else  
            tmp[7]=0;     
          //计算LBP编码  
            s.val[0]=(tmp[0]*1+tmp[1]*2+tmp[2]*4+tmp[3]*8+tmp[4]*16+tmp[5]*32+tmp[6]*64+tmp[7]*128);  
          
        cvSet2D(dst,i,j,s);  
      }  
}  


  


}