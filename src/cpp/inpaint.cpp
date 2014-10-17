#include "photoAlgo.h"
//#include "inpaint.h"

namespace customCV {

#define KNOWN  0  //known outside narrow band
#define BAND   1  //narrow band (known)
#define INSIDE 128  //unknown
#define CHANGE 255  //servise

	typedef struct CvHeapElem
	{
		float T;
		int i, j;
		struct CvHeapElem* prev;
		struct CvHeapElem* next;
	} CvHeapElem;


	class CvPriorityQueueFloat
	{
	protected:
		CvHeapElem *mem, *empty, *head, *tail;
		int num, in;

	public:
		bool Init(const CvMat* f)
		{
			int i, j;
			for (i = num = 0; i < f->rows; i++)
			{
				for (j = 0; j < f->cols; j++)
					num += CV_MAT_ELEM(*f, uchar, i, j) != 0;
			}
			if (num <= 0) return false;
			mem = (CvHeapElem*)cvAlloc((num + 2)*sizeof(CvHeapElem));
			if (mem == NULL) return false;

			head = mem;
			head->i = head->j = -1;
			head->prev = NULL;
			head->next = mem + 1;
			head->T = -FLT_MAX;
			empty = mem + 1;
			for (i = 1; i <= num; i++) {
				mem[i].prev = mem + i - 1;
				mem[i].next = mem + i + 1;
				mem[i].i = -1;
				mem[i].T = FLT_MAX;
			}
			tail = mem + i;
			tail->i = tail->j = -1;
			tail->prev = mem + i - 1;
			tail->next = NULL;
			tail->T = FLT_MAX;
			return true;
		}


		bool Add(const CvMat* f) {  //一个包装函数，对插入已满进行判断
			int i, j;
			for (i = 0; i < f->rows; i++) {
				for (j = 0; j < f->cols; j++) {
					if (CV_MAT_ELEM(*f, uchar, i, j) != 0) {
						if (!Push(i, j, 0)) return false;      //队列已经满了
					}
				}
			}
			return true;
		}

		bool Push(int i, int j, float T) {   //这种push，pop效率太低
			CvHeapElem *tmp = empty, *add = empty;
			if (empty == tail) return false;
			while (tmp->prev->T > T) tmp = tmp->prev;   //按照T值排序。
			if (tmp != empty) {
				add->prev->next = add->next;
				add->next->prev = add->prev;
				empty = add->next;
				add->prev = tmp->prev;
				add->next = tmp;
				add->prev->next = add;
				add->next->prev = add;
			}
			else {
				empty = empty->next;
			}
			add->i = i;
			add->j = j;
			add->T = T;
			in++;
			//      printf("push i %3d  j %3d  T %12.4e  in %4d\n",i,j,T,in);
			return true;
		}

		bool Pop(int *i, int *j) {
			CvHeapElem *tmp = head->next;
			if (empty == tmp) return false;
			*i = tmp->i;
			*j = tmp->j;
			tmp->prev->next = tmp->next;
			tmp->next->prev = tmp->prev;
			tmp->prev = empty->prev;
			tmp->next = empty;
			tmp->prev->next = tmp;
			tmp->next->prev = tmp;
			empty = tmp;
			in--;
			//      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
			return true;
		}

		bool Pop(int *i, int *j, float *T) {
			CvHeapElem *tmp = head->next;
			if (empty == tmp) return false;
			*i = tmp->i;
			*j = tmp->j;
			*T = tmp->T;
			tmp->prev->next = tmp->next;
			tmp->next->prev = tmp->prev;
			tmp->prev = empty->prev;
			tmp->next = empty;
			tmp->prev->next = tmp;
			tmp->next->prev = tmp;
			empty = tmp;
			in--;
			//      printf("pop  i %3d  j %3d  T %12.4e  in %4d\n",tmp->i,tmp->j,tmp->T,in);
			return true;
		}

		CvPriorityQueueFloat(void) {
			num = in = 0;
			mem = empty = head = tail = NULL;
		}

		~CvPriorityQueueFloat(void)
		{
			cvFree(&mem);
		}
	};

	inline float VectorScalMult(CvPoint2D32f v1, CvPoint2D32f v2) {
		return v1.x*v2.x + v1.y*v2.y;
	}

	inline float VectorLength(CvPoint2D32f v1) {
		return v1.x*v1.x + v1.y*v1.y;
	}

	template <typename T>
	T min(T a, T b) {
		if (a < b)
			return a;
		else
			return b;
	}

#define MAT_ELEM( mat, elemtype, row, col )           \
	*(mat.data + mat.step * row + sizeof(elemtype)* col)

#define MAT_ELEM3( mat, elemtype, row, col, color)           \
	*(mat.data + mat.step * row + sizeof(elemtype)* (col * 3 + color))

	Inpaint::Params::Params(int range) {
		range_ = range;
	}

	class InpaintFmmImpl : public Inpaint {
	public:
		InpaintFmmImpl(const Params& params) {
			setParams(params);
		}
		virtual ~InpaintFmmImpl() {}
		void setParams(const Params& params) {
			params_ = params;
		}
		Params getParams() const {
			return params_;
		}

		cv::Mat apply(cv::Mat src, cv::InputArray maskMat = cv::noArray()) {
			cv::Mat dst;
			cv::Mat inpaint_mask;
			inpaint_mask = maskMat.getMat();
			int range = params_.range_;
			range = max(range, 1);
			range = min(range, 100);
			int ecols, erows;
			ecols = src.cols + 2;
			erows = src.rows + 2;
			src.copyTo(dst);
			cv::Mat f(erows, ecols, CV_8UC1, cv::Scalar(KNOWN, 0, 0, 0));
			cv::Mat mask;
			mask.create(erows, ecols, CV_8UC1);
			cv::Mat band(erows, ecols, CV_8UC1, cv::Scalar(KNOWN, 0, 0, 0));
			cv::Mat t(erows, ecols, CV_32FC1, cv::Scalar(1.0e6f, 0, 0, 0));
			cv::Mat el_cross, el_range;
			el_cross = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
			copyMaskBorder(inpaint_mask, mask);
			setBorder(mask, 0);
			cv::dilate(mask, band, el_cross);
			cv::Ptr<CvPriorityQueueFloat> Heap, Out;

			Heap = new CvPriorityQueueFloat;
			if (!Heap->Init(&(CvMat(band))))
				return dst;

			cv::subtract(band, mask, band);
			setBorder(band, 0);
			if (!Heap->Add(&(CvMat(band))))
				return dst;
			cvSet(&(CvMat(f)), cvScalar(BAND, 0, 0, 0), &(CvMat(band)));
			cvSet(&(CvMat(f)), cvScalar(INSIDE, 0, 0, 0), &(CvMat(mask)));
			cvSet(&(CvMat(t)), cvScalar(0, 0, 0, 0), &(CvMat(band)));

			cv::Mat o;
			o.create(erows, ecols, CV_8UC1);
			el_range = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(2 * range + 1, 2 * range + 1), cv::Point(range, range));
			cv::dilate(mask, o, el_range);
			cv::subtract(o, mask, o);
			Out = new CvPriorityQueueFloat;
			if (!Out->Init(&(CvMat(o))))
				return dst;
			if (!Out->Add(&(CvMat(band))))
				return dst;

			cv::subtract(o, band, o);
			setBorder(o, 0);

			std::cout << "fmm" << std::endl;
			fmm(o, t, Out, true);
			std::cout << "inpaintfmm" << std::endl;
			inpaintFmm(mask, t, dst, range, Heap);
			return dst;
		}


		
		template<typename T>
		T getElement(T t, const cv::Mat& mat, int row, int col, int color) {
			return *(mat.data + mat.step * row + sizeof(T)* (col * mat.channels() + color));
		}

		template <typename T>
		T max(T a, T b) {
			if (a > b)
				return a;
			else
				return b;
		}

		void copyMaskBorder(cv::Mat src, cv::Mat dst) {
			for (int i = 0; i < src.rows; i++) {
				for (int j = 0; j < src.cols; j++) {
					if (src.at<uchar>(i, j) != 0) {
						dst.at<uchar>(i, j) = INSIDE;
					}
				}
			}
		}

		void setBorder(cv::Mat image, int value) {
			int i, j;
			int cols = image.cols, rows = image.rows;
			for (j = 0; j < cols; j++) {
				image.at<uchar>(0, j) = value;
			}
			for (i = 0; i < rows; i++) {
				image.at<uchar>(i, 0) = image.at<uchar>(i, cols - 1) = value;
			}
			for (j = 0; j < cols; j++) {
				image.at<uchar>(rows - 1, j) = value;
			}
		}

		float solve(int i1, int j1, int i2, int j2, cv::Mat& f, cv::Mat& t) {
			float ret;
			float t1 = t.at<float>(i1, j1);
			float t2 = t.at<float>(i2, j2);
			float min_t = min(t1, t2);
			int f1 = MAT_ELEM(f, uchar, i1, j1);
			int f2 = MAT_ELEM(f, uchar, i2, j2);
			if (f1 != INSIDE) {
				if (f2 != INSIDE) {
					if (fabs(t1 - t2) >= 0.1)
						ret = 1 + min_t;
					else
						ret = (t1 + t2 + sqrt((double)(2 - (t1 - t2)*(t1 - t2)))) * 0.5;
				}
				else {
					ret = 1 + t1;
				}
			}
			else if (f2 != INSIDE) {
				ret = 1 + t2;
			}
			else {
				ret = 1 + min_t;
			}
			return ret;
		}

		void fmm(cv::Mat f, cv::Mat t, CvPriorityQueueFloat *Heap, bool negate) {
			int pop_i, pop_j;
			int dir_i[] = { -1, 0, 1, 0 };
			int dir_j[] = { 0, -1, 0, 1 };
			while (Heap->Pop(&pop_i, &pop_j)) {
				unsigned char known = (negate) ? CHANGE : KNOWN;
				f.at<uchar>(pop_i, pop_j) = known;

				for (int dir = 0; dir < 4; dir++) {
					int i = pop_i + dir_i[dir];
					int j = pop_j + dir_j[dir];
					if (i <= 0 || j <= 0 || i > f.rows || j > f.cols) {
						continue;
					}

					if (MAT_ELEM(f, uchar, i, j) == INSIDE) {
						float dist;
						float temp1, temp2, temp3, temp4;
						temp1 = solve(i - 1, j, i, j - 1, f, t);
						temp2 = solve(i + 1, j, i, j - 1, f, t);
						temp3 = solve(i - 1, j, i, j + 1, f, t);
						temp4 = solve(i + 1, j, i, j + 1, f, t);
						float d1 = min(temp1, temp2);
						float d2 = min(temp3, temp4);
						dist = min(d1, d2);

						t.at<float>(i, j) = dist;
						f.at<uchar>(i, j) = BAND;
						Heap->Push(i, j, dist);
					}
				}
			}

			if (negate) {
				for (int i = 0; i < f.rows; i++) {
					for (int j = 0; j < f.cols; j++) {
						if (MAT_ELEM(f, uchar, i, j) == CHANGE) {
							f.at<uchar>(i, j) = KNOWN;
							t.at<float>(i, j) = -1 * t.at<float>(i, j);
						}

					}
				}
			}
		}

		cv::Point2f clacGradT(cv::Mat f, cv::Mat t, int i, int j) {
			cv::Point2f gradT;
			uchar lhs, rhs, mid;
			lhs = MAT_ELEM(f, uchar, i, j + 1);
			rhs = MAT_ELEM(f, uchar, i, j - 1);
			mid = MAT_ELEM(f, uchar, i, j);
			if (lhs != INSIDE) {
				if (rhs != INSIDE) {
					gradT.x = (lhs - rhs) * 0.5;
				}
				else {
					gradT.x = lhs - mid;
				}
			}
			else {
				if (rhs != INSIDE) {
					gradT.x = mid - rhs;
				}
				else {
					gradT.x = 0;
				}
			}

			lhs = MAT_ELEM(f, uchar, i + 1, j);
			rhs = MAT_ELEM(f, uchar, i - 1, j);
			mid = MAT_ELEM(f, uchar, i, j);
			if (lhs != INSIDE) {
				if (rhs != INSIDE) {
					gradT.y = (lhs - rhs) * 0.5;
				}
				else {
					gradT.y = lhs - mid;
				}
			}
			else {
				if (rhs != INSIDE) {
					gradT.y = mid - rhs;
				}
				else {
					gradT.y = 0;
				}
			}

			return gradT;
		}

		cv::Point2f clacGradI3(cv::Mat& f, cv::Mat& dst_img, int x, int y, int xp, int xm, int yp, int ym, int color) {
			CvPoint2D32f gradI;
			uchar lhs, rhs, mid, mid2;
			lhs = MAT_ELEM3(dst_img, uchar, ym, xp + 1, color);
			rhs = MAT_ELEM3(dst_img, uchar, ym, xm - 1, color);
			mid = MAT_ELEM3(dst_img, uchar, ym, xm, color);
			mid2 = MAT_ELEM3(dst_img, uchar, ym, xp, color);
			if (MAT_ELEM(f, uchar, y, x + 1) != INSIDE) {
				if (MAT_ELEM(f, uchar, y, x - 1) != INSIDE) {
					gradI.x = (lhs - rhs) * 2.0;
				}
				else {
					gradI.x = lhs - mid;
				}
			}
			else {
				if (MAT_ELEM(f, uchar, y, x - 1) != INSIDE) {
					gradI.x = mid2 - rhs;
				}
				else {
					gradI.x = 0;
				}
			}

			lhs = MAT_ELEM3(dst_img, uchar, yp + 1, xm, color);
			rhs = MAT_ELEM3(dst_img, uchar, ym - 1, xm, color);
			mid = MAT_ELEM3(dst_img, uchar, ym, xm, color);
			mid2 = MAT_ELEM3(dst_img, uchar, yp, xm, color);
			if (MAT_ELEM(f, uchar, y + 1, x) != INSIDE) {
				if (MAT_ELEM(f, uchar, y - 1, x) != INSIDE) {
					gradI.y = (lhs - rhs) * 2.0;
				}
				else {
					gradI.y = lhs - mid;
				}
			}
			else {
				if (MAT_ELEM(f, uchar, y - 1, x) != INSIDE) {
					gradI.y = mid2 - rhs;
				}
				else {
					gradI.y = 0;
				}
			}

			return gradI;
		}


		cv::Point2f clacGradI(cv::Mat f, cv::Mat t, int y, int x, int ym, int xm, int yp, int xp) {
			cv::Point2f gradI;
			if (f.at<uchar>(y, x + 1) != INSIDE) {
				if (f.at<uchar>(y, x - 1) != INSIDE) {
					gradI.x = (t.at<float>(yp + 1, xm) - t.at<float>(ym - 1, xm)) * 2.0;
				}
				else {
					gradI.x = t.at<float>(yp + 1, xm) - t.at<float>(ym, xm);
				}
			}
			else {
				if (f.at<uchar>(y, x - 1) != INSIDE) {
					gradI.x = t.at<float>(ym, xp) - t.at<float>(ym, xm - 1);
				}
				else {
					gradI.x = 0;
				}
			}

			if (f.at<uchar>(y + 1, x) != INSIDE) {
				if (f.at<uchar>(y - 1, x) != INSIDE) {
					gradI.y = (t.at<float>(yp + 1, xm) - t.at<float>(ym - 1, xm)) * 2.0;
				}
				else {
					gradI.y = t.at<float>(yp + 1, xm) - t.at<float>(ym, xm);
				}
			}
			else {
				if (f.at<uchar>(y - 1, x) != INSIDE) {
					gradI.y = t.at<float>(yp, xm) - t.at<float>(ym - 1, xm);
				}
				else {
					gradI.y = 0;
				}
			}
			return gradI;
		}


		float vectorLength(cv::Point2f p) {
			return p.x * p.x + p.y * p.y;
		}

		float vectorScalMult(cv::Point2f p1, cv::Point2f p2) {
			return p1.x * p2.x + p1.y * p2.y;
		}

		void inpaintFmm(cv::Mat f, cv::Mat t, cv::Mat& dst_img, int range, CvPriorityQueueFloat *Heap) {
			int range2 = range * range;
			int pop_i, pop_j;
			int dir_i[] = { -1, 0, 1, 0 };
			int dir_j[] = { 0, -1, 0, 1 };
			while (Heap->Pop(&pop_i, &pop_j)) {
				f.at<uchar>(pop_i, pop_j) = KNOWN;

				for (int dirIndex = 0; dirIndex < 4; dirIndex++) {
					int i = pop_i + dir_i[dirIndex];
					int j = pop_j + dir_j[dirIndex];
					if (i <= 0 || j <= 0 || i > f.rows || j > f.cols) {
						continue;
					}

					//梯度计算
					if (f.at<uchar>(i, j) == INSIDE) {
						float dist;
						float temp1, temp2, temp3, temp4;
						temp1 = solve(i - 1, j, i, j - 1, f, t);
						temp2 = solve(i + 1, j, i, j - 1, f, t);
						temp3 = solve(i - 1, j, i, j + 1, f, t);
						temp4 = solve(i + 1, j, i, j + 1, f, t);
						float d1 = min(temp1, temp2);
						float d2 = min(temp3, temp4);
						dist = min(d1, d2);

						t.at<float>(i, j) = dist;

						cv::Point2f gradI, gradT, r;
						float Ia[4] = { 0 }, Jx[4] = { 0 }, Jy[4] = { 0 }, s[4] = { 1.0e-20f, 1.0e-20f, 1.0e-20f, 1.0e-20f }, w, sat;
						gradT = clacGradT(f, t, i, j);

						for (int y = i - range; y <= i + range; y++) {
							int ym = y - 1 + (y == 1); // [1,+inf] 
							int yp = y - 1 - (y == t.rows - 2);  //[-inf, t.rows-3], 以上两句保证边缘有个padding
							for (int x = j - range; x < j + range; x++) {
								int xm = x - 1 + (x == 1);
								int xp = x - 1 - (x == t.cols - 2);
								if (y < 0 || x < 0 || y > t.rows - 1 || x > t.cols - 1 ||
									MAT_ELEM(f, uchar, y, x) == INSIDE || (x - j)*(x - j) + (y - i)*(y - i) > range2) {
									continue;
								}

								r.y = i - y;
								r.x = j - x;

								float dst = 1. / (VectorLength(r) * sqrt(VectorLength(r)));
								float lev = t.at<float>(y, x) - t.at<float>(i, j);
								if (lev < 0) {
									lev = -lev;
								}
								lev = 1. / (1 + lev);

								float dir = vectorScalMult(r, gradT);

								if (dir >= -0.01 && dir <= 0.01)
									dir = 0.0000001f;
								float w = dst * lev * dir;
								if (w < 0) {
									w = -w;
								}

								if (dst_img.channels() == 3) {
									for (int color = 0; color <= 2; color++) {
										if (f.at<uchar>(y, x + 1) != INSIDE) {
											if (f.at<uchar>(y, x - 1) != INSIDE) {
												gradI.x = (dst_img.at<cv::Vec3b>(yp + 1, xm)[color] - dst_img.at<cv::Vec3b>(ym - 1, xm)[color]) * 2.0f;
											}
											else {
												gradI.x = dst_img.at<cv::Vec3b>(yp + 1, xm)[color] - dst_img.at<cv::Vec3b>(ym, xm)[color];
											}
										}
										else {
											if (f.at<uchar>(y, x - 1) != INSIDE) {
												gradI.x = dst_img.at<cv::Vec3b>(ym, xp)[color] - dst_img.at<cv::Vec3b>(ym, xm - 1)[color];
											}
											else {
												gradI.x = 0;
											}
										}

										//heat 3
										if (f.at<uchar>(y + 1, x) != INSIDE) {
											if (f.at<uchar>(y - 1, x) != INSIDE) {
												gradI.y = (dst_img.at<cv::Vec3b>(yp + 1, xm)[color] - dst_img.at<cv::Vec3b>(ym - 1, xm)[color]) * 2.0;
											}
											else {
												gradI.y = dst_img.at<cv::Vec3b>(yp + 1, xm)[color] - dst_img.at<cv::Vec3b>(ym, xm)[color];
											}
										}
										else {
											if (f.at<uchar>(y - 1, x) != INSIDE) {
												gradI.y = dst_img.at<cv::Vec3b>(yp, xm)[color] - dst_img.at<cv::Vec3b>(ym - 1, xm)[color];
											}
											else {
												gradI.y = 0;
											}
										}

										Ia[color] += (float)w * MAT_ELEM3(dst_img, uchar, ym, xm, color);
										Jx[color] -= (float)w * (float)(gradI.x * r.x);
										Jy[color] -= (float)w * (float)(gradI.y * r.y);
										s[color] += w;
									}
								}
								else if(dst_img.channels() == 1) {
									if (f.at<uchar>(y, x + 1) != INSIDE) {
										if (f.at<uchar>(y, x - 1) != INSIDE) {
											gradI.x = (dst_img.at<uchar>(yp + 1, xm) - dst_img.at<uchar>(ym - 1, xm)) * 2.0f;
										}
										else {
											gradI.x = dst_img.at<uchar>(yp + 1, xm) - dst_img.at<uchar>(ym, xm);
										}
									}
									else {
										if (f.at<uchar>(y, x - 1) != INSIDE) {
											gradI.x = dst_img.at<uchar>(ym, xp) - dst_img.at<uchar>(ym, xm - 1);
										}
										else {
											gradI.x = 0;
										}
									}

									if (f.at<uchar>(y + 1, x) != INSIDE) {
										if (f.at<uchar>(y - 1, x) != INSIDE) {
											gradI.y = (dst_img.at<uchar>(yp + 1, xm) - dst_img.at<uchar>(ym - 1, xm)) * 2.0;
										}
										else {
											gradI.y = dst_img.at<uchar>(yp + 1, xm) - dst_img.at<uchar>(ym, xm);
										}
									}
									else {
										if (f.at<uchar>(y - 1, x) != INSIDE) {
											gradI.y = dst_img.at<uchar>(yp, xm) - dst_img.at<uchar>(ym - 1, xm);
										}
										else {
											gradI.y = 0;
										}
									}

									Ia[0] += (float)w * MAT_ELEM(dst_img, uchar, ym, xm);
									Jx[0] -= (float)w * (float)(gradI.x * r.x);
									Jy[0] -= (float)w * (float)(gradI.y * r.y);
									s[0] += w;
								}
							} // end for x
						} // end for y

						if (dst_img.channels() == 3) {
							for (int color = 0; color <= 2; color++) {
								sat = (float)((Ia[color] / s[color] + (Jx[color] + Jy[color]) / (sqrt(Jx[color] * Jx[color] + Jy[color] * Jy[color]) + 1.0e-20f) + 0.5f));
								dst_img.at<cv::Vec3b>(i - 1, j - 1)[color] = cv::saturate_cast<uchar>(sat);
							}
						} 
						else if (dst_img.channels() == 1) {
							sat = (float)((Ia[0] / s[0] + (Jx[0] + Jy[0]) / (sqrt(Jx[0] * Jx[0] + Jy[0] * Jy[0]) + 1.0e-20f) + 0.5f));
							dst_img.at<uchar>(i - 1, j - 1) = cv::saturate_cast<uchar>(sat);
						}
						f.at<uchar>(i, j) = BAND;
						Heap->Push(i, j, dist);
					}
				}
			}
		}

	public:
		Params params_;
	};

	cv::Ptr<Inpaint> Inpaint::create(int type, const Params& params) {
		if (type == INPAINT_FMM) {
			return cv::Ptr<Inpaint>(new InpaintFmmImpl(params));
		}
	}


}