#pragma once

#include <dlib/opencv.h>
#include "BFM.h"
#include "Renderer.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include "ceres/cubic_interpolation.h"

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

struct SparseCost {
	SparseCost(BFM bfm_, Vector2d observed_landmark_, int vertex_id_, int image_width_, int image_height_) :
		bfm{bfm_}, observed_landmark{observed_landmark_}, vertex_id{vertex_id_}, image_width{image_width_}, image_height{image_height_}
	{}
	template <typename T>
	bool operator()(T const* camera_rotation, T const* camera_translation, T const* shape_weights, T const* exp_weights, T* residuals) const {
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> shape_pca_basis_full(bfm.shape_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> exp_pca_basis_full(bfm.exp_pca_basis, 85764, 100);

		Matrix<T, 1, 3> face_model;

		face_model(0, 0) = T(bfm.shape_mean[vertex_id * 3]) + T(bfm.exp_mean[vertex_id * 3]);
		face_model(0, 1) = T(bfm.shape_mean[vertex_id * 3 + 1]) + T(bfm.exp_mean[vertex_id * 3 + 1]);
		face_model(0, 2) = T(bfm.shape_mean[vertex_id * 3 + 2]) + T(bfm.exp_mean[vertex_id * 3 + 2]);

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.shape_pca_var[i])) * shape_weights[i];
			face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(shape_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(shape_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 100; i++) {
			T value = T(sqrt(bfm.exp_pca_var[i])) * exp_weights[i];
			face_model(0, 0) += T(exp_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(exp_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(exp_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}
				
		Matrix<T, 3, 1> translation = { camera_translation[0], camera_translation[1], camera_translation[2] };
		Quaternion<T> rotation = { camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3] };
		
		auto transformation_matrix = calculate_transformation_matrix<T>(translation, rotation);
		auto projection = calculate_transformation_perspective<T>(image_width, image_height, transformation_matrix, face_model);
		
		T w = projection(0, 3);
		T transformed_x = (projection(0, 0) / w + T(1)) / T(2)* T(image_width);
		T transformed_y = (projection(0, 1) / w + T(1)) / T(2)* T(image_height);
		transformed_y = T(image_height) - transformed_y;
		
		residuals[0] = transformed_x - T(observed_landmark[0]);
		residuals[1] = transformed_y - T(observed_landmark[1]);
		
		return true;
	}
private:
	const BFM bfm;
	const Vector2d observed_landmark;
	const int vertex_id;
	const int image_width;
	const int image_height;
};

struct DenseRGBCost {
	DenseRGBCost(BFM bfm_, cv::Mat* image_, int vertex_id_) :
		bfm{ bfm_ }, image{ image_ }, vertex_id{ vertex_id_ }
	{}
	template <typename T>
	bool operator()(T const* camera_rotation, T const* camera_translation, T const* shape_weights, T const* exp_weights, T const* color_weights, T* residuals) const {
		int image_width = image->cols;
		int image_height = image->rows;
		
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> shape_pca_basis_full(bfm.shape_pca_basis, 85764, 199);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> exp_pca_basis_full(bfm.exp_pca_basis, 85764, 100);
		Map<Matrix<double, Dynamic, Dynamic, RowMajor>> color_pca_basis_full(bfm.color_pca_basis, 85764, 199);

		Matrix<T, 1, 3> face_model;
		Matrix<T, 1, 3> face_color;

		face_model(0, 0) = T(bfm.shape_mean[vertex_id * 3]) + T(bfm.exp_mean[vertex_id * 3]);
		face_model(0, 1) = T(bfm.shape_mean[vertex_id * 3 + 1]) + T(bfm.exp_mean[vertex_id * 3 + 1]);
		face_model(0, 2) = T(bfm.shape_mean[vertex_id * 3 + 2]) + T(bfm.exp_mean[vertex_id * 3 + 2]);

		face_color(0, 0) = T(bfm.color_mean[vertex_id * 3]);
		face_color(0, 1) = T(bfm.color_mean[vertex_id * 3 + 1]);
		face_color(0, 2) = T(bfm.color_mean[vertex_id * 3 + 2]);

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.shape_pca_var[i])) * shape_weights[i];
			face_model(0, 0) += T(shape_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(shape_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(shape_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 100; i++) {
			T value = T(sqrt(bfm.exp_pca_var[i])) * exp_weights[i];
			face_model(0, 0) += T(exp_pca_basis_full(vertex_id * 3, i)) * value;
			face_model(0, 1) += T(exp_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_model(0, 2) += T(exp_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		for (int i = 0; i < 199; i++) {
			T value = T(sqrt(bfm.color_pca_var[i])) * color_weights[i];
			face_color(0, 0) += T(color_pca_basis_full(vertex_id * 3, i)) * value;
			face_color(0, 1) += T(color_pca_basis_full(vertex_id * 3 + 1, i)) * value;
			face_color(0, 2) += T(color_pca_basis_full(vertex_id * 3 + 2, i)) * value;
		}

		Matrix<T, 3, 1> translation = { camera_translation[0], camera_translation[1], camera_translation[2] };
		Quaternion<T> rotation = { camera_rotation[0], camera_rotation[1], camera_rotation[2], camera_rotation[3] };

		auto transformation_matrix = calculate_transformation_matrix<T>(translation, rotation);
		auto projection = calculate_transformation_perspective<T>(image_width, image_height, transformation_matrix, face_model);

		T w = projection(0, 3);
		T transformed_x = (projection(0, 0) / w + T(1)) / T(2) * T(image_width);
		T transformed_y = (projection(0, 1) / w + T(1)) / T(2) * T(image_height);
		transformed_y = T(image_height) - transformed_y;

		ceres::Grid2D<uchar, 3> grid(image->ptr(0), 0, image->rows, 0, image->cols);
		ceres::BiCubicInterpolator<ceres::Grid2D<uchar, 3>> interpolator(grid);
		T observed_colour[3];
		interpolator.Evaluate(transformed_y, transformed_x, &observed_colour[0]);

		residuals[0] = face_color(0, 0) * 255.0 - T(observed_colour[2]);
		residuals[1] = face_color(0, 1) * 255.0 - T(observed_colour[1]);
		residuals[2] = face_color(0, 2) * 255.0 - T(observed_colour[0]);

		return true;
	}
private:
	const BFM bfm;
	const cv::Mat* image;
	const int vertex_id;
};

class DenseOptimizer {
public:
	void optimize(cv::Mat image, std::vector<dlib::full_object_detection> detected_landmarks);
private: 
	void render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation);
	int render_number;
};