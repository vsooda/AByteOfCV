// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include "full_object_detection.h"
#include "../algs.h"
#include "../matrix.h"
#include "../pixel.h"
#include <opencv2/core/core.hpp>
#include "common.h"

#define LANDMARK_NUM  68
#define CASCADE_NUM 15
#define TREE_PER_CASCADE 500
#define LEAF_NUM 16 
#define SPLIT_NUM 15


struct AffineTransform {
	cv::Mat_<float> rotation;
	float scale;
	AffineTransform(cv::Mat_<float> rotation_, float scale_) {
		//rotation = rotation_;
		rotation_.copyTo(rotation);
		scale = scale_;
	}
	cv::Mat getRotation() {
		return rotation;
	}
};


namespace dlib
{

	float pointDistance(cv::Point2f lhs, cv::Point2f rhs) {
		cv::Point2f temp = lhs - rhs;
		return temp.x * temp.x + temp.y * temp.y;
	}
// ----------------------------------------------------------------------------------------
	struct ShapeLandmark {
		std::vector<float> positions;
	};

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

    // ------------------------------------------------------------------------------------
		/*
        inline point_transform_affine find_tform_between_shapes (
            const cv::Mat& from_shape,
            const cv::Mat& to_shape
        )
        {
			//std::cout << "shape size " << from_shape.size() << " " << to_shape.size() << std::endl;
            //CV_ASSERT(from_shape.size() == to_shape.size() && (from_shape.size()%2) == 0 && from_shape.size() > 0);
//            std::vector<cv::Point2f > from_points, to_points;
			std::vector<vector<float, 2> > from_points, to_points;
			const int num = LANDMARK_NUM;
            from_points.reserve(num);
            to_points.reserve(num);
            if (num == 1)
            {
                // Just use an identity transform if there is only one landmark.
                return point_transform_affine();
            }

            for (int i = 0; i < num; ++i)
            {
				//from_points.push_back(location(from_shape, i));
                //to_points.push_back(location(to_shape,i));
				from_points.push_back(vector<float, 2>(from_shape.at<float>(i * 2, 0), from_shape.at<float>(i * 2 + 1, 0)));
				to_points.push_back(vector<float, 2>(to_shape.at<float>(i * 2, 0), to_shape.at<float>(i * 2 + 1, 0)));
            }
            return find_similarity_transform(from_points, to_points);
        }

		*/

		cv::Mat point2Mat(const cv::Point& pt, const rectangle& rect) {
			cv::Mat ptMat(2, 1, CV_32FC1);
			ptMat.at<float>(0, 0) = pt.x;
			ptMat.at<float>(1, 0) = pt.y;
			return ptMat;
		}
		cv::Mat_<float> ProjectShape(const cv::Mat_<float>& shape, const rectangle& bounding_box){
			cv::Mat_<float> temp(shape.rows, 1);
			int width = bounding_box.right() - bounding_box.left();
			int height = bounding_box.bottom() - bounding_box.top();
			int ptNum = shape.rows / 2; 
			for (int j = 0; j < ptNum;j++){
				temp(j*2, 0) = (shape(j*2, 0) - bounding_box.left()) / width ;
				temp(j*2+1, 1) = (shape(j*2+1, 0) - bounding_box.top()) / height;
			}
			return temp;
		}

		cv::Mat_<float> ReProjectShape(const cv::Mat_<float>& shape, const rectangle& bounding_box){
			cv::Mat_<float> temp(shape.rows, 1);
			int width = bounding_box.right() - bounding_box.left();
			int height = bounding_box.bottom() - bounding_box.top();
			int ptNum = shape.rows / 2; 
			for (int j = 0; j < ptNum; j++){
				temp(2*j, 0) = (shape(2 * j, 0) * width + bounding_box.left());
				temp(2*j+1, 0) = (shape(2 * j + 1, 0) * height + bounding_box.top());
			} 
			return temp;
		}

		AffineTransform SimilarityTransform(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2){
			cv::Mat_<float> rotation;
			float scale;
			rotation = cv::Mat::zeros(2, 2, CV_32FC1);
			scale = 0;

			// center the data
			double center_x_1 = 0;
			double center_y_1 = 0;
			double center_x_2 = 0;
			double center_y_2 = 0;
			for (int i = 0; i < shape1.rows; i++){
				center_x_1 += shape1(i, 0);
				center_y_1 += shape1(i, 1);
				center_x_2 += shape2(i, 0);
				center_y_2 += shape2(i, 1);
			}
			center_x_1 /= shape1.rows;
			center_y_1 /= shape1.rows;
			center_x_2 /= shape2.rows;
			center_y_2 /= shape2.rows;

			cv::Mat_<double> temp1 = shape1.clone();
			cv::Mat_<double> temp2 = shape2.clone();
			for (int i = 0; i < shape1.rows; i++){
				temp1(i, 0) -= center_x_1;
				temp1(i, 1) -= center_y_1;
				temp2(i, 0) -= center_x_2;
				temp2(i, 1) -= center_y_2;
			}


			cv::Mat_<double> covariance1, covariance2;
			cv::Mat_<double> mean1, mean2;
			// calculate covariance matrix
			calcCovarMatrix(temp1, covariance1, mean1, CV_COVAR_COLS);
			calcCovarMatrix(temp2, covariance2, mean2, CV_COVAR_COLS);

			double s1 = cv::sqrt(norm(covariance1));
			double s2 = cv::sqrt(norm(covariance2));
			scale = s1 / s2;
			temp1 = 1.0 / s1 * temp1;
			temp2 = 1.0 / s2 * temp2;

			double num = 0;
			double den = 0;
			for (int i = 0; i < shape1.rows; i++){
				num = num + temp1(i, 1) * temp2(i, 0) - temp1(i, 0) * temp2(i, 1);
				den = den + temp1(i, 0) * temp2(i, 0) + temp1(i, 1) * temp2(i, 1);
			}

			double norm = cv::sqrt(num*num + den*den);
			double sin_theta = num / norm;
			double cos_theta = den / norm;
			rotation(0, 0) = cos_theta;
			rotation(0, 1) = -sin_theta;
			rotation(1, 0) = sin_theta;
			rotation(1, 1) = cos_theta;
			return AffineTransform(rotation, scale);
		}

		AffineTransform SimilarityTransform1(const cv::Mat_<float>& shape1, const cv::Mat_<float>& shape2){
			cv::Mat_<float> rotation;
			float scale;
			rotation = cv::Mat::zeros(2, 2, CV_32FC1);
			scale = 0;

			// center the data
			double center_x_1 = 0;
			double center_y_1 = 0;
			double center_x_2 = 0;
			double center_y_2 = 0;

			int ptNum = shape1.rows / 2;

			for (int i = 0; i < ptNum; i++){
				center_x_1 += shape1(2*i, 0);
				center_y_1 += shape1(2*i+1, 0);
				center_x_2 += shape2(2*i, 0);
				center_y_2 += shape2(2*i+1, 0);
			}
			center_x_1 /= ptNum;
			center_y_1 /= ptNum;
			center_x_2 /= ptNum;
			center_y_2 /= ptNum;

			cv::Mat_<double> temp1 = shape1.clone();
			cv::Mat_<double> temp2 = shape2.clone();
			for (int i = 0; i < ptNum; i++){
				temp1(2*i, 0) -= center_x_1;
				temp1(2*i+1, 0) -= center_y_1;
				temp2(2*i, 0) -= center_x_2;
				temp2(2*i+1, 0) -= center_y_2;
			}


			cv::Mat_<double> covariance1, covariance2;
			cv::Mat_<double> mean1, mean2;
			// calculate covariance matrix
			calcCovarMatrix(temp1, covariance1, mean1, CV_COVAR_COLS);
			calcCovarMatrix(temp2, covariance2, mean2, CV_COVAR_COLS);

			double s1 = cv::sqrt(norm(covariance1));
			double s2 = cv::sqrt(norm(covariance2));
			scale = s1 / s2;
			temp1 = 1.0 / s1 * temp1;
			temp2 = 1.0 / s2 * temp2;

			double num = 0;
			double den = 0;
			for (int i = 0; i < ptNum; i++){
				num = num + temp1(2*i+1, 0) * temp2(2*i, 0) - temp1(2*i, 0) * temp2(2*i+1, 0);
				den = den + temp1(2*i, 0) * temp2(2*i, 0) + temp1(2*i+1, 0) * temp2(2*i+1, 0);
			}

			double norm = cv::sqrt(num*num + den*den);
			double sin_theta = num / norm;
			double cos_theta = den / norm;
			rotation(0, 0) = cos_theta;
			rotation(0, 1) = -sin_theta;
			rotation(1, 0) = sin_theta;
			rotation(1, 1) = cos_theta;
			return AffineTransform(rotation, scale);
		}


    // ------------------------------------------------------------------------------------
		/*
        inline point_transform_affine normalizing_tform (
            const rectangle& rect
        )
        {
            std::vector<vector<float,2> > from_points, to_points;
            from_points.push_back(rect.tl_corner()); to_points.push_back(point(0,0));
            from_points.push_back(rect.tr_corner()); to_points.push_back(point(1,0));
            from_points.push_back(rect.br_corner()); to_points.push_back(point(1,1));
            return find_similarity_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine unnormalizing_tform (
            const rectangle& rect
        )
        {
            std::vector<vector<float,2> > from_points, to_points;
            to_points.push_back(rect.tl_corner()); from_points.push_back(point(0,0));
            to_points.push_back(rect.tr_corner()); from_points.push_back(point(1,0));
            to_points.push_back(rect.br_corner()); from_points.push_back(point(1,1));
            return find_similarity_transform(from_points, to_points);
        }
		*/
    // ------------------------------------------------------------------------------------

		void extract_feature_pixel_values(
			cv::Mat img,
            const rectangle& rect,
            const cv::Mat& current_shape,
            const cv::Mat& reference_shape,
            const std::vector<int>& reference_pixel_anchor_idx,
            const std::vector<cv::Point2f >& reference_pixel_deltas,
            std::vector<float>& feature_pixel_values
        )
        {
            //const matrix<float,2,2> tform = matrix_cast<float>(find_tform_between_shapes(reference_shape, current_shape).get_m());
			cv::Mat tform = SimilarityTransform1(reference_shape, current_shape).getRotation();
            //const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            feature_pixel_values.resize(reference_pixel_deltas.size());
            for (int i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                //point p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
				cv::Mat tempDelta = point2Mat(reference_pixel_deltas[i], rect);
				cv::Mat locateMat = point2Mat(location(current_shape, reference_pixel_anchor_idx[i]), rect);
				cv::Mat pointMat = ReProjectShape(tform * tempDelta + locateMat, rect);
				cv::Point p;
				p.x = pointMat.at<float>(0, 0);
				p.y = pointMat.at<float>(1, 0);
				if (p.x <= img.cols || p.y <= img.rows)
					feature_pixel_values[i] = img.at<float>(p.y, p.x);
                else
                    feature_pixel_values[i] = 0;
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
            const rectangle& rect
        ) const
        {
            using namespace impl;
			cv::Mat current_shape;
			initial_shape.copyTo(current_shape);
            std::vector<float> feature_pixel_values;
            for (int iter = 0; iter < forests.size(); ++iter)
            {
				//std::cout << "current_shape " << current_shape.size() << "  initial_shape " << initial_shape.size() << std::endl;
                extract_feature_pixel_values(img, rect, current_shape, initial_shape, anchor_idx[iter], deltas[iter], feature_pixel_values);
				//std::cout << "current_shape " << current_shape.size() << "  initial_shape " << initial_shape.size() << std::endl;
                // evaluate all the trees at this level of the cascade.
				for (int i = 0; i < forests[iter].size(); ++i) {
					//std::cout << "forest: " << forests[iter][i](feature_pixel_values).size() << std::endl;
					current_shape += forests[iter][i](feature_pixel_values);
				}
            }

			std::cout << "current_shape_size: " << current_shape.size() << std::endl;
			std::cout << "before project " << current_shape << std::endl;
			cv::Mat imgShape = ReProjectShape(current_shape, rect);
			std::cout << "detect" << imgShape << std::endl;
			std::cout << "rect " << rect << std::endl;
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
						//std::cout << i << " " << idx << " " <<  deltas[i][idx](1);
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

#endif // DLIB_SHAPE_PREDICToR_H_

