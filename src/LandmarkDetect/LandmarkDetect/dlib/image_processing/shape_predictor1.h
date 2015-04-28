// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_1
#define DLIB_SHAPE_PREDICToR_H_1

#include "shape_predictor_abstract.h"
#include "full_object_detection.h"
#include "../algs.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../pixel.h"
#include <opencv2/core/core.hpp>

#define LANDMARK_NUM  74
using dlib::matrix;
using dlib::rectangle;
using dlib::vector;
using dlib::point_transform_affine;
using dlib::point;
using dlib::full_object_detection;

namespace dlib1
{

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

            friend inline void serialize (const split_feature& item, std::ostream& out)
            {
                dlib::serialize(item.idx1, out);
                dlib::serialize(item.idx2, out);
                dlib::serialize(item.thresh, out);
            }
            friend inline void deserialize (split_feature& item, std::istream& in)
            {
                dlib::deserialize(item.idx1, in);
                dlib::deserialize(item.idx2, in);
                dlib::deserialize(item.thresh, in);
            }
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
            std::vector<matrix<float,0,1> > leaf_values;

            inline const matrix<float,0,1>& operator()(
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

            friend void serialize (const regression_tree& item, std::ostream& out)
            {
                dlib::serialize(item.splits, out);
                dlib::serialize(item.leaf_values, out);
            }
            friend void deserialize (regression_tree& item, std::istream& in)
            {
                dlib::deserialize(item.splits, in);
                dlib::deserialize(item.leaf_values, in);
            }
			void read(const cv::FileNode& node) {
				assert(node.type() == cv::FileNode::MAP);
				splits = std::vector<split_feature>(15);
				cv::FileNode split_nodes = node["split"];
				//leaf_values = std::vector<matrix<float, LANDMARK_NUM * 2, 1> >(16);
				for (int i = 0; i < 16; i++) {
					matrix<float, 0, 1> temp;
					temp.set_size(LANDMARK_NUM * 2, 1);
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
				for (int i = 0; i < leafMat.cols; i++) { //16列
					//matrix<float, 0, 1> temp;
					//temp.set_size(LANDMARK_NUM * 2, 1);
					for (int j = 0; j < leafMat.rows; j++) {
						leaf_values[i](j) = leafMat.at<float>(j, i);
						//temp(j, 1) = leafMat.at<float>(j, i);
					}
					//leaf_values.push_back(temp);
					//leaf_values[i] = temp;
				}
					
			}

			void write(cv::FileStorage& fs) const {
				assert(fs.isOpened());
				fs << "{" << "split" << "[";
				for (int i = 0; i < 15; i++) {
					fs  << splits[i];
				}
				fs << "]";
				cv::Mat leafMat(LANDMARK_NUM * 2, 16, CV_32FC1);
				for (int i = 0; i < leafMat.cols; i++) {
					for (int j = 0; j < leafMat.rows; j++) {
						//每个叶子一列
						leafMat.at<float>(j, i) = leaf_values[i](j);
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

        inline vector<float,2> location (
            const matrix<float,0,1>& shape,
            int idx
        )
        {
            return vector<float,2>(shape(idx*2), shape(idx*2+1));
        }

    // ------------------------------------------------------------------------------------

        inline int nearest_shape_point (
            const matrix<float,0,1>& shape,
            const dlib::vector<float,2>& pt
        )
        {
            // find the nearest part of the shape to this pixel
            float best_dist = std::numeric_limits<float>::infinity();
            const int num_shape_parts = shape.size()/2;
            int best_idx = 0;
            for (int j = 0; j < num_shape_parts; ++j)
            {
                const float dist = length_squared(location(shape,j)-pt);
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
			const matrix<float, 0, 1>& shape,
			const std::vector<dlib::vector<float, 2> >& pixel_coordinates,
            std::vector<int>& anchor_idx, 
            std::vector<dlib::vector<float,2> >& deltas
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

		template <typename T>
		point_transform_affine find_similarity_transform1(
			const std::vector<dlib::vector<T, 2> >& from_points,
			const std::vector<dlib::vector<T, 2> >& to_points
			)
		{
			
			int ptNum = from_points.size();
			double dptNum = ptNum * 1.0;
			cv::Mat_<double> shape1(2, ptNum);
			cv::Mat_<double> shape2(2, ptNum);
			for (int i = 0; i < ptNum; i++) {
				shape1(0, i) = from_points[i].x();
				shape1(1, i) = from_points[i].y();
				shape2(0, i) = to_points[i].x();
				shape2(1, i) = to_points[i].y();
			}
			cv::Mat_<double> rotation;
			float scale;
			rotation = cv::Mat::zeros(2, 2, CV_64FC1);
			scale = 0;

			//cv::Mat_<double> mean_from(2, 1, cv::Scalar(0.0)), mean_to(2, 1, cv::Scalar(0.0));
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
			//std::cout << mean_from << std::endl;
			//std::cout << mean_to << std::endl;
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
			//matlab µÄsvdºÍÕâÀïµÄw²»ÏàÍ¬ http://stackoverflow.com/questions/12029486/matlab-svd-output-in-opencv
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
			cv::Mat t = mean_to - c * r * mean_from;
			
			r = c * r;

			dlib::matrix<double, 2, 2> dlib_r;
			dlib::vector<double, 2> dlib_t;
			for (int i = 0; i < 2; i++) { //co;
				for (int j = 0; j < 2; j++) { //row
					dlib_r(i, j) = r.at<double>(j, i);
				}
			}
			for (int i = 0; i < 2; i++) {
				dlib_t(i) = t.at<double>(i, 0);
			}
			//std::cout << dlib_r << dlib_t << std::endl;
			std::cout << dlib_r << std::endl;

			return point_transform_affine(dlib_r, dlib_t);
		}

        inline point_transform_affine find_tform_between_shapes (
            const dlib::matrix<float,0,1>& from_shape,
            const dlib::matrix<float,0,1>& to_shape
        )
        {
			//std::cout << "shape size " << from_shape.size() << " " << to_shape.size() << std::endl;
            DLIB_ASSERT(from_shape.size() == to_shape.size() && (from_shape.size()%2) == 0 && from_shape.size() > 0,"");
            std::vector<vector<float,2> > from_points, to_points;
            const int num = from_shape.size()/2;
            from_points.reserve(num);
            to_points.reserve(num);
            if (num == 1)
            {
                // Just use an identity transform if there is only one landmark.
                return point_transform_affine();
            }

            for (int i = 0; i < num; ++i)
            {
                from_points.push_back(location(from_shape,i));
                to_points.push_back(location(to_shape,i));
            }
            return find_similarity_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------------------

        void extract_feature_pixel_values (
            const cv::Mat& img,
            const rectangle& rect,
            const matrix<float,0,1>& current_shape,
            const matrix<float,0,1>& reference_shape,
            const std::vector<int>& reference_pixel_anchor_idx,
            const std::vector<dlib::vector<float,2> >& reference_pixel_deltas,
            std::vector<float>& feature_pixel_values
        )
        {
            const matrix<float,2,2> tform = dlib::matrix_cast<float>(find_tform_between_shapes(reference_shape, current_shape).get_m());
			//std::cout << tform << std::endl;
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            feature_pixel_values.resize(reference_pixel_deltas.size());
            for (int i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                point p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
				/*std::cout << " ---------" << std::endl;
				std::cout << reference_pixel_deltas[i] << std::endl;
				std::cout << location(current_shape, reference_pixel_anchor_idx[i]) << std::endl;
				std::cout << tform << std::endl;
				std::cout << tform_to_img.get_m() << std::endl;
				std::cout << tform_to_img.get_b() << std::endl;
				std::cout << "p: " << p << std::endl;*/
				if (p.y() < img.rows && p.x() < img.cols)
					//feature_pixel_values[i] = get_pixel_intensity(img[p.y()][p.x()]);
					feature_pixel_values[i] = img.at<float>(p.y(), p.x());
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
        shape_predictor (
        ) 
        {}

        shape_predictor (
            const matrix<float,0,1>& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,
            const std::vector<std::vector<dlib::vector<float,2> > >& pixel_coordinates
        ) : initial_shape(initial_shape_), forests(forests_)
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());
            // Each cascade uses a different set of pixels for its features.  We compute
            // their representations relative to the initial shape now and save it.
            for (int i = 0; i < pixel_coordinates.size(); ++i)
                impl::create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
        }

        int num_parts (
        ) const
        {
            return initial_shape.size()/2;
        }

		dlib::full_object_detection operator()(
            const cv::Mat& img,
            const rectangle& rect
        ) const
        {
            using namespace impl;
			matrix<float, 0, 1> current_shape;
			current_shape.set_size(LANDMARK_NUM * 2, 1);
			current_shape = initial_shape;
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

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);
            std::vector<point> parts(current_shape.size()/2);
            for (int i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
			std::cout << "tform " << std::endl;
			std::cout << tform_to_img.get_m() << std::endl;
			std::cout << tform_to_img.get_b() << std::endl;
			/*cv::Mat det(LANDMARK_NUM, 2, CV_32F);
			for (int i = 0; i < LANDMARK_NUM; i++) {
			det.at<float>(i, 0) = parts[i].x();
			det.at<float>(i, 1) = parts[i].y();
			}
			return det;*/
            return dlib::full_object_detection(rect, parts);
        }

        friend void serialize (const shape_predictor& item, std::ostream& out)
        {
            int version = 1;
            dlib::serialize(version, out);
            dlib::serialize(item.initial_shape, out);
            dlib::serialize(item.forests, out);
            dlib::serialize(item.anchor_idx, out);
            dlib::serialize(item.deltas, out);
        }
        friend void deserialize (shape_predictor& item, std::istream& in)
        {
            int version = 0;
            dlib::deserialize(version, in);
            dlib::deserialize(item.initial_shape, in);
            dlib::deserialize(item.forests, in);
            dlib::deserialize(item.anchor_idx, in);
            dlib::deserialize(item.deltas, in);
        }

		void read(const cv::FileNode& node) {
			assert(node.type() == cv::FileNode::MAP);
			//cv::Mat shape0(LANDMARK_NUM * 2, 1, CV_32FC1); 
			cv::Mat shape0;
			node["init_shape"] >> shape0;
			initial_shape.set_size(LANDMARK_NUM * 2, 1);
			//dlib::matrix<float, LANDMARK_NUM * 2, 1> temp;
			//initial_shape = temp;
			for (int i = 0; i < LANDMARK_NUM * 2; i++) {
				initial_shape(i) = shape0.at<float>(i, 0);
			}
			std::cout << "initial_shape:: " << initial_shape.size() << std::endl;
			forests = std::vector<std::vector<impl::regression_tree> >(15);
			for (int i = 0; i < 15; i++) {
				forests[i] = std::vector<impl::regression_tree>(500);
			}
			anchor_idx = std::vector<std::vector<int> >(15);
			for (int i = 0; i < 15; i++) {
				anchor_idx[i] = std::vector<int>(500);
			}
			deltas = std::vector<std::vector<dlib::vector<float, 2> > >(15);
			for (int i = 0; i < 15; i++) {
				deltas[i] = std::vector<dlib::vector<float, 2> >(500);
			}
			
			char forest_name[50];
			for (int i = 0; i < 15; i++) {
				sprintf(forest_name, "forest_name_%03d", i);
				cv::FileNode forest_node = node[forest_name];
				cv::FileNodeIterator it = forest_node.begin(), it_end = forest_node.end();
				int idx = 0;
				for (; it != it_end; ++it, idx++) {
						(*it) >> forests[i][idx];
				}
			}

			char anchor_name[50];
			for (int i = 0; i < 15; i++) {
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
			for (int i = 0; i < 15; i++) {
				sprintf(delta_name, "delta_name_%03d", i);
				cv::FileNode delta_node = node[delta_name];
				cv::FileNodeIterator it = delta_node.begin(), it_end = delta_node.end();
				int idx = 0;
				std::cout << i << std::endl;
				for (; it != it_end; ++it, idx++) {
						(*it)["delta_x"] >> deltas[i][idx](0);
						(*it)["delta_y"] >> deltas[i][idx](1);
						//std::cout << i << " " << idx << " " <<  deltas[i][idx](1);
				}
			}
			std::cout << "shape over" << std::endl;


		}

		void write(cv::FileStorage& fs) const {
			assert(fs.isOpened());
			cv::Mat shape0(LANDMARK_NUM * 2, 1, CV_32FC1);
			fs << "{";
			for (int i = 0; i < LANDMARK_NUM * 2; i++) {
				shape0.at<float>(i, 0) = initial_shape(i);
			}
			fs << "init_shape" << shape0;
			char forest_name[50];
			for (int i = 0; i < 15; i++) {
				sprintf(forest_name, "forest_name_%03d", i);
				fs << forest_name << "[";
				for (int j = 0; j < 500; j++) {
					fs << forests[i][j];
				}
				fs << "]";
			}

			char anchor_name[50];
			for (int i = 0; i < 15; i++) {
				sprintf(anchor_name, "anchor_idx_%03d", i);
				fs << anchor_name << "[";
				for (int j = 0; j < 500; j++) {
					fs  << anchor_idx[i][j];
				}
				fs << "]";
			}

			char delta_name[50];
			for (int i = 0; i < 15; i++) {
				sprintf(delta_name, "delta_name_%03d", i);
				fs << delta_name << "[";
				for (int j = 0; j < 500; j++) {
					fs << "{" << "delta_x" << deltas[i][j](0) << "delta_y" << deltas[i][j](1) << "}";
				}
				fs << "]";
			}
		}

    private:
        matrix<float,0,1> initial_shape;
        std::vector<std::vector<impl::regression_tree> > forests;
        std::vector<std::vector<int> > anchor_idx; 
        std::vector<std::vector<dlib::vector<float,2> > > deltas;
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

