#pragma once 
#include <opencv2/core/core.hpp>
#include "common.h"

#define LANDMARK_NUM  68
#define CASCADE_NUM 15
#define TREE_PER_CASCADE 500
#define LEAF_NUM 16 
#define SPLIT_NUM 15


struct AffineTransform {
	cv::Mat_<float> rotation;
	cv::Mat_<float> b;
	AffineTransform(cv::Mat_<float> rotation_, cv::Mat_<float> b_) {
		rotation_.copyTo(rotation);
		b_.copyTo(b);
	}
	cv::Mat getRotation() {
		return rotation;
	}
	cv::Mat getB() {
		return b;
	}

	cv::Mat operator()(const cv::Mat& locateMat) {
		cv::Mat ret;
		ret = rotation * locateMat;
		for (int i = 0; i < locateMat.cols; i++) {
			ret.at<float>(0, i) = ret.at<float>(0, i) + b.at<float>(0, 0);
		}
		for (int i = 0; i < locateMat.cols; i++) {
			ret.at<float>(1, i) = ret.at<float>(1, i) + b.at<float>(1, 0);
		}
		return ret;
	}
};


namespace customCV
{

	float pointDistance(cv::Point2f lhs, cv::Point2f rhs) {
		cv::Point2f temp = lhs - rhs;
		return temp.x * temp.x + temp.y * temp.y;
	}
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct split_feature
        {
			int idx1;
			int idx2;
            float thresh;

			void read(const cv::FileNode& node) {
				assert(node.type() == cv::FileNode::MAP);
				node["idx1"] >> idx1;
				node["idx2"] >> idx2;
				node["thresh"] >> thresh;
			}

			void write(cv::FileStorage& fs) const {
				assert(fs.isOpened());
				fs << "{"
					<< "idx1" << idx1
					<< "idx2" << idx2
					<< "thresh" << thresh
					<< "}";
			}

        };
		static void write(cv::FileStorage& fs, const std::string&, const split_feature& x)
		{
			x.write(fs);
		}
		static void read(const cv::FileNode& node, split_feature& x, const split_feature& default_value = split_feature())
		{
			if (node.empty())
				x = default_value;
			else
				x.read(node);
		}


        // a tree is just a std::vector<impl::split_feature>.  We use this function to navigate the
        // tree nodes
        inline int left_child (int idx) { return 2*idx + 1; }
       
        inline int right_child (int idx) { return 2*idx + 2; }
       

        struct regression_tree
        {
            std::vector<split_feature> splits;
            std::vector<cv::Mat > leaf_values;

            inline const cv::Mat& operator()(
                const std::vector<float>& feature_pixel_values
            ) const
            {
                int i = 0;
                while (i < splits.size())
                {
                    if (feature_pixel_values[splits[i].idx1] - feature_pixel_values[splits[i].idx2] > splits[i].thresh)
                        i = left_child(i);
                    else
                        i = right_child(i);
                }
                return leaf_values[i - splits.size()];
            }

			void read(const cv::FileNode& node) {
				assert(node.type() == cv::FileNode::MAP);
				splits = std::vector<split_feature>(SPLIT_NUM);
				cv::FileNode split_nodes = node["split"];
				//leaf_values = std::vector<matrix<float, 136, 1> >(LEAF_NUM);
				for (int i = 0; i < LEAF_NUM; i++) {
					cv::Mat temp(LANDMARK_NUM * 2, 1, CV_32FC1);
					leaf_values.push_back(temp);
				}
				cv::FileNodeIterator it = split_nodes.begin(), it_end = split_nodes.end();
				int idx = 0;
				for (; it != it_end; ++it, idx++) {
					(*it) >> splits[idx];
				}

				//std::cout << "reading leaf" << std::endl;
				cv::Mat leafMat;
				node["leaft_values"] >> leafMat;
				//std::cout << leafMat.size() << std::endl;
				for (int i = 0; i < leafMat.cols; i++) { //LEAF_NUM列
					//matrix<float, 0, 1> temp;
					//temp.set_size(136, 1);
					for (int j = 0; j < leafMat.rows; j++) {
						leaf_values[i].at<float>(j, 0) = leafMat.at<float>(j, i);
						//temp(j, 1) = leafMat.at<float>(j, i);
					}
					//leaf_values.push_back(temp);
					//leaf_values[i] = temp;
				}
					
			}

			void write(cv::FileStorage& fs) const {
				assert(fs.isOpened());
				fs << "{" << "split" << "[";
				for (int i = 0; i < SPLIT_NUM; i++) {
					fs  << splits[i];
				}
				fs << "]";
				cv::Mat leafMat(LANDMARK_NUM * 2, LEAF_NUM, CV_32FC1);
				for (int i = 0; i < leafMat.cols; i++) {
					for (int j = 0; j < leafMat.rows; j++) {
						//每个叶子一列
						//leafMat.at<float>(j, i) = leaf_values[i](j);
						leafMat.at<float>(j, i) = leaf_values[i].at<float>(j, 0);
					}
				}
				fs << "leaft_values" << leafMat;
				fs << "}";
			}
			
        };

		static void write(cv::FileStorage& fs, const std::string&, const regression_tree& x)
		{
			x.write(fs);
		}

		static void read(const cv::FileNode& node, regression_tree& x, const regression_tree& default_value = regression_tree())
		{
			if (node.empty())
				x = default_value;
			else
				x.read(node);
		}

    // ------------------------------------------------------------------------------------

       // inline vector<float,2> location (const cv::Mat& shape,int idx) {
       //     return vector<float,2>(shape.at<float>(idx*2, 0), shape.at<float>(idx*2+1, 0));
       // }
		inline cv::Point2f location(const cv::Mat& shape, int idx) {
			return cv::Point2f(shape.at<float>(idx * 2, 0), shape.at<float>(idx * 2 + 1, 0));
		}

    // ------------------------------------------------------------------------------------
		

        inline int nearest_shape_point (const cv::Mat& shape, const cv::Point2f& pt) {
            // find the nearest part of the shape to this pixel
            float best_dist = std::numeric_limits<float>::infinity();
			const int num_shape_parts = LANDMARK_NUM;
            int best_idx = 0;
            for (int j = 0; j < num_shape_parts; ++j)
            {
				const float dist = pointDistance(location(shape, j), pt);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            return best_idx;
        }

    // ------------------------------------------------------------------------------------
		//每一层cascade获得这个anchor_idx,delta
		inline void create_shape_relative_encoding(
			const cv::Mat& shape,
			const std::vector<cv::Point2f>& pixel_coordinates,
            std::vector<int>& anchor_idx, 
            std::vector<cv::Point2f >& deltas
        )
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());


            for (int i = 0; i < pixel_coordinates.size(); ++i)
            {
                anchor_idx[i] = nearest_shape_point(shape, pixel_coordinates[i]);
                deltas[i] = pixel_coordinates[i] - location(shape,anchor_idx[i]);
            }
        }


		cv::Mat point2Mat(const cv::Point2f& pt) { //cv::Point& pt is wrong !!!
			cv::Mat ptMat(2, 1, CV_32FC1);
			ptMat.at<float>(0, 0) = pt.x;
			ptMat.at<float>(1, 0) = pt.y;
			return ptMat;
		}

		AffineTransform SimilarityTransform(const cv::Mat_<float>& sp1, const cv::Mat_<float>& sp2){
			int ptNum = sp1.rows / 2;
			double dptNum = ptNum * 1.0;
			cv::Mat_<double> shape1(2, ptNum);
			cv::Mat_<double> shape2(2, ptNum);
			for (int i = 0; i < ptNum; i++) {
				shape1(0, i) = sp1(2*i, 0);
				shape1(1, i) = sp1(2*i+1, 0);
				shape2(0, i) = sp2(2*i, 0);
				shape2(1, i) = sp2(2*i+1, 0);
			}
			cv::Mat_<double> rotation;
			float scale;
			rotation = cv::Mat::zeros(2, 2, CV_64FC1);
			scale = 0;

			cv::Mat_<double> mean_from = cv::Mat(2, 1, CV_64FC1, cv::Scalar(0));
			cv::Mat_<double> mean_to = cv::Mat(2, 1, CV_64FC1, cv::Scalar(0));

			cv::Mat_<double> cov = cv::Mat::zeros(2, 2, CV_64FC1);
			for (int i = 0; i < ptNum; i++) {
				mean_from(0, 0) += shape1(0, i);
				mean_from(1, 0) += shape1(1, i);
				mean_to(0, 0) += shape2(0, i);
				mean_to(1, 0) += shape2(1, i);
			}

			mean_from = mean_from / dptNum;
			mean_to = mean_to / dptNum;
			double sigma_from = 0, sigma_to = 0;
			for (int i = 0; i < ptNum; i++) {
				double diff1 = (shape1(0, i) - mean_from(0, 0)) * (shape1(0, i) - mean_from(0, 0)) + (shape1(1, i) - mean_from(1, 0)) * (shape1(1, i) - mean_from(1, 0));
				double diff2 = (shape2(0, i) - mean_to(0, 0)) * (shape2(0, i) - mean_to(0, 0)) + (shape2(1, i) - mean_to(1, 0)) * (shape2(1, i) - mean_to(1, 0));
				sigma_from += diff1;
				sigma_to += diff2;
				cv::Mat_<double> temp(2, 2);
				cv::Mat_<double> d1(2, 1), d2(2, 1);
				d1(0, 0) = shape1(0, i) - mean_from(0, 0);
				d1(1, 0) = shape1(1, i) - mean_from(1, 0);
				d2(0, 0) = shape2(0, i) - mean_to(0, 0);
				d2(1, 0) = shape2(1, i) - mean_to(1, 0);
				temp = d1 * d2.t();
				cov = cov + temp;
			}

			sigma_from = sigma_from / dptNum;
			sigma_to = sigma_to / dptNum;
			cov = cov / dptNum;
			//std::cout << sigma_from << std::endl;
			//std::cout << sigma_to << std::endl;
			cv::SVD svd;
			cv::Mat u, vt, d, s, w;
			//matlab diff form opencv http://stackoverflow.com/questions/12029486/matlab-svd-output-in-opencv
			//svd.compute(cov, d, u, vt, cv::SVD::FULL_UV);
			cv::SVDecomp(cov, w, u, vt, cv::SVD::FULL_UV);
			d = cv::Mat(cov.size(), CV_64FC1, cv::Scalar(0));
			for (int i = 0; i < w.rows; i++) {
				d.at<double>(i, i) = w.at<double>(i, 0);
			}
			//std::cout << std::endl;
			//std::cout << cov << std::endl;
			//std::cout << u << std::endl;
			//std::cout << d << std::endl;
			//std::cout << vt << std::endl;
			s = cv::Mat::eye(cov.size(), CV_64FC1);

			if (cv::determinant(cov) < 0) {
				if (d.at<double>(1, 1) < d.at<double>(0, 0))
					s.at<double>(1, 1) = -1;
				else
					s.at<double>(0, 0) = -1;
			}

			cv::Mat r(2, 2, CV_64FC1);
			r = u * s * vt;
			double c = 1;
			if (sigma_from != 0) {
				c = 1.0 / sigma_from * cv::trace(d * s)[0];
			}
			
			r = c * r;
			r = r.t();
			cv::Mat t = mean_to - r * mean_from;

			std::cout << r << std::endl << std::endl;;

			t.convertTo(t, CV_32F);
			r.convertTo(r, CV_32F);
			return AffineTransform(r, t);
		}

    // ------------------------------------------------------------------------------------
		
		inline AffineTransform normalizing_tform(
            const cv::Rect& rect
        )
        {
			cv::Mat_<float>sp1 = cv::Mat(6, 1, CV_32F);
			cv::Mat_<float>sp2 = cv::Mat(6, 1, CV_32F);
			float left = rect.x;
			float right = rect.x + rect.width;
			float top = rect.y;
			float bottom = rect.y + rect.height;
			sp1(0, 0) = left;
			sp1(1, 0) = top;
			sp1(2, 0) = right;
			sp1(3, 0) = top;
			sp1(4, 0) = right;
			sp1(5, 0) = bottom;
			sp2(0, 0) = 0;
			sp2(1, 0) = 0;
			sp2(2, 0) = 1;
			sp2(3, 0) = 0;
			sp2(4, 0) = 1;
			sp2(5, 0) = 1;
			return SimilarityTransform(sp1, sp2);
        }

    // ------------------------------------------------------------------------------------

		inline AffineTransform unnormalizing_tform(
            const cv::Rect& rect
        )
        {
			cv::Mat_<float>sp1 = cv::Mat(6, 1, CV_32F);
			cv::Mat_<float>sp2 = cv::Mat(6, 1, CV_32F);
			float left = rect.x;
			float right = rect.x + rect.width;
			float top = rect.y;
			float bottom = rect.y + rect.height;
			sp1(0, 0) = left;
			sp1(1, 0) = top;
			sp1(2, 0) = right;
			sp1(3, 0) = top;
			sp1(4, 0) = right;
			sp1(5, 0) = bottom;
			sp2(0, 0) = 0;
			sp2(1, 0) = 0;
			sp2(2, 0) = 1;
			sp2(3, 0) = 0;
			sp2(4, 0) = 1;
			sp2(5, 0) = 1;
			return SimilarityTransform(sp2, sp1);
        }
    // ------------------------------------------------------------------------------------

		void extract_feature_pixel_values(
			cv::Mat img,
            const cv::Rect& rect,
            const cv::Mat& current_shape,
            const cv::Mat& reference_shape,
            const std::vector<int>& reference_pixel_anchor_idx,
            const std::vector<cv::Point2f >& reference_pixel_deltas,
            std::vector<float>& feature_pixel_values
        )
        {
            //const matrix<float,2,2> tform = matrix_cast<float>(find_tform_between_shapes(reference_shape, current_shape).get_m());
			cv::Mat tform = SimilarityTransform(reference_shape, current_shape).getRotation();
            //const point_transform_affine tform_to_img = unnormalizing_tform(rect);
			AffineTransform tform_to_img = unnormalizing_tform(rect);
			/*for (int i = 0; i < 50; i++) {
				std::cout << reference_pixel_deltas[i].x << " " << reference_pixel_deltas[i].y << std::endl;
				}*/

            feature_pixel_values.resize(reference_pixel_deltas.size());
            for (int i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                //point p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
				cv::Mat tempDelta = point2Mat(reference_pixel_deltas[i]);
				cv::Mat locateMat = point2Mat(location(current_shape, reference_pixel_anchor_idx[i]));
				//cv::Mat pointMat = ReProjectShape(tform * tempDelta + locateMat, rect);
				cv::Mat pointMat = tform_to_img(tform * tempDelta + locateMat);
				cv::Point p;
				p.x = pointMat.at<float>(0, 0);
				p.y = pointMat.at<float>(1, 0);
				/*std::cout << feature_pixel_values[i] << std::endl;
				std::cout << " ---------" << std::endl;
				std::cout << tempDelta << std::endl;
				std::cout << locateMat << std::endl;
				std::cout << tform << std::endl;
				std::cout << tform_to_img.getRotation() << std::endl;
				std::cout << tform_to_img.getB() << std::endl;
				std::cout << "p: " << p << std::endl;
				if (i == 5)
				exit(0);*/
				if (p.x < img.cols && p.y < img.rows)
					feature_pixel_values[i] = img.at<float>(p.y, p.x);
				else
					feature_pixel_values[i] = 0;
            }
			for (int i = 0; i < 10; i++) {
				std::cout << feature_pixel_values[i] << std::endl;
			}
        }


    } // end namespace impl

// ----------------------------------------------------------------------------------------

    class shape_predictor
    {
    public:
        shape_predictor () {}

        shape_predictor (
            const cv::Mat& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,
            const std::vector<std::vector<cv::Point2f > >& pixel_coordinates
        ) : initial_shape(initial_shape_), forests(forests_)
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());
            // Each cascade uses a different set of pixels for its features.  We compute
            // their representations relative to the initial shape now and save it.
            for (int i = 0; i < pixel_coordinates.size(); ++i)
                impl::create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
        }


        cv::Mat operator()(
            cv::Mat& img,
            const cv::Rect& rect
        ) const
        {
            using namespace impl;
			cv::Mat current_shape;
			initial_shape.copyTo(current_shape);
            std::vector<float> feature_pixel_values;
            for (int iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);
                // evaluate all the trees at this level of the cascade.
				for (int i = 0; i < forests[iter].size(); ++i) {
					current_shape += forests[iter][i](feature_pixel_values);
				}
            }

			AffineTransform tform_to_img = unnormalizing_tform(rect);
			cv::Mat currentMat(2, LANDMARK_NUM, CV_32F);
			for (int i = 0; i < LANDMARK_NUM; i++) {
				currentMat.at<float>(0, i) = current_shape.at<float>(2 * i, 0);
				currentMat.at<float>(1, i) = current_shape.at<float>(2 * i + 1, 0);
			}
			std::cout << "tform_to_img "<< std::endl;
			std::cout << tform_to_img.getB().size() << std::endl;
			std::cout << tform_to_img.getB() << std::endl;

			cv::Mat imgShape = tform_to_img(currentMat);
			return imgShape;
        }

		void read(const cv::FileNode& node) {
			assert(node.type() == cv::FileNode::MAP);
			node["init_shape"] >> initial_shape;
			std::cout << "initial_shape:: " << initial_shape.size() << std::endl;
			forests = std::vector<std::vector<impl::regression_tree> >(CASCADE_NUM);
			for (int i = 0; i < CASCADE_NUM; i++) {
				forests[i] = std::vector<impl::regression_tree>(TREE_PER_CASCADE);
			}
			anchor_idx = std::vector<std::vector<int> >(CASCADE_NUM);
			for (int i = 0; i < CASCADE_NUM; i++) {
				anchor_idx[i] = std::vector<int>(TREE_PER_CASCADE);
			}
			deltas = std::vector<std::vector<cv::Point2f> >(CASCADE_NUM);
			for (int i = 0; i < CASCADE_NUM; i++) {
				deltas[i] = std::vector<cv::Point2f>(TREE_PER_CASCADE);
			}
			
			char forest_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(forest_name, "forest_name_%03d", i);
				cv::FileNode forest_node = node[forest_name];
				cv::FileNodeIterator it = forest_node.begin(), it_end = forest_node.end();
				int idx = 0;
				for (; it != it_end; ++it, idx++) {
						(*it) >> forests[i][idx];
				}
			}

			char anchor_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(anchor_name, "anchor_idx_%03d", i);
				cv::FileNode anchor_node = node[anchor_name];
				cv::FileNodeIterator it = anchor_node.begin(), it_end = anchor_node.end();
				int idx = 0;
				for (; it != it_end; ++it, idx++) {
					//std::cout << anchor_name << " " << idx << std::endl;
						(*it) >> anchor_idx[i][idx];
				}
			}
			std::cout << "anchor over" << std::endl;

			char delta_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(delta_name, "delta_name_%03d", i);
				cv::FileNode delta_node = node[delta_name];
				cv::FileNodeIterator it = delta_node.begin(), it_end = delta_node.end();
				int idx = 0;
				std::cout << i << std::endl;
				for (; it != it_end; ++it, idx++) {
						(*it)["delta_x"] >> deltas[i][idx].x;
						(*it)["delta_y"] >> deltas[i][idx].y;
				}
			}
			std::cout << "shape over" << std::endl;


		}

		void write(cv::FileStorage& fs) const {
			assert(fs.isOpened());
			fs << "{";
			fs << "init_shape" << initial_shape;
			char forest_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(forest_name, "forest_name_%03d", i);
				fs << forest_name << "[";
				for (int j = 0; j < TREE_PER_CASCADE; j++) {
					fs << forests[i][j];
				}
				fs << "]";
			}

			char anchor_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(anchor_name, "anchor_idx_%03d", i);
				fs << anchor_name << "[";
				for (int j = 0; j < TREE_PER_CASCADE; j++) {
					fs  << anchor_idx[i][j];
				}
				fs << "]";
			}

			char delta_name[50];
			for (int i = 0; i < CASCADE_NUM; i++) {
				sprintf(delta_name, "delta_name_%03d", i);
				fs << delta_name << "[";
				for (int j = 0; j < TREE_PER_CASCADE; j++) {
					fs << "{" << "delta_x" << deltas[i][j].x << "delta_y" << deltas[i][j].y << "}";
				}
				fs << "]";
			}
		}

    private:
        cv::Mat initial_shape;
        std::vector<std::vector<impl::regression_tree> > forests;
        std::vector<std::vector<int> > anchor_idx; 
        std::vector<std::vector<cv::Point2f > > deltas;
    };

	static void write(cv::FileStorage& fs, const std::string&, const shape_predictor& x)
	{
		x.write(fs);
	}

	static void read(const cv::FileNode& node, shape_predictor& x, const shape_predictor& default_value = shape_predictor())
	{
		if (node.empty())
			x = default_value;
		else
			x.read(node);
	}

}


