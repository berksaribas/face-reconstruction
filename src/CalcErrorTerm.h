#pragma once
#include <vector>
#include <FacialLandmarkDetection.h>

#include "utils/points.h"

#include "ceres/ceres.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Eigen.h"
//#include "ProcrustesAligner.h"
#include <BFM.h>

struct SparseCostFunction
{
	SparseCostFunction(const Point3D& point1_, const Weight& weight_)
		: point1(point1_), weight(weight_)
	{
	}

	template<typename T>
	bool operator()(const T* Transform, const T* parameters, T* residual) const
	{
		//auto cosinus = cos(angle[0]);
		//auto sinus = sin(angle[0]);
		//Point2D p_transformed = point1;
		//T trans_x = cosinus * point1.x - sinus * point1.y + tx[0];
		//T trans_y = sinus * point1.x + cosinus * point1.y + ty[0];
		auto res1 = weight.w * (point1.x - point2.x);
		auto res2 = weight.w * (point1.y - point2.y);
		auto res2 = weight.w * (point1.z - point2.z);
		residual[0] = T(res1);
		residual[1] = T(res2);
		residual[1] = T(res2);
		return true;
	}

private:
	const Point3D point1;
	const Weight weight;
};

float CalcSparseTerm(std::vector<int> model_landmarks, Parameters params, std::vector<dlib::full_object_detection> rgb_landmarks) {
	// model_landmarks are the indices of the vertices corresponsing to the landmarks
	std::cout << model_landmarks.size() << std::endl;
	for (int i = 0; i < model_landmarks.size(); i++)
	{

		//std::cout << landmark << std::endl;
	}
	return 0.;
}

Matrix4f GetPose(BFM bfm, Parameters params, std::vector<dlib::full_object_detection> rgb_landmarks) {
	// TODO: convert model_landmarks and rgb_landmarks to vector<vector3f> types
	
	std::vector<Eigen::Vector3f> model_landmarks_vec3;
	std::vector<Eigen::Vector3f> rgb_landmarks_vec3;

	for (int i = 0; i < bfm.landmarks.size(); i++) {
		MatrixXf shape_pca_var = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
		MatrixXf shape_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

		MatrixXf exp_pca_var = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
		MatrixXf exp_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);
		MatrixXf shape_result = shape_pca_basis * (shape_pca_var * params.shape_weights);
		MatrixXf exp_result = exp_pca_basis * (exp_pca_var * params.exp_weights);
		MatrixXf result = shape_result + exp_result;
		int land_idx = bfm.landmarks[i];
		//float vx = bfm.shape_mean[land_idx] + bfm.exp_mean[land_idx] + result(land_idx);
		//float vy = bfm.shape_mean[land_idx +1] + bfm.exp_mean[land_idx +1] + result(land_idx +1);
		//float vz = bfm.shape_mean[land_idx +2] + bfm.exp_mean[land_idx +2] + result(land_idx +2);
		//Vector3f v1(vx, vy, vz);
		//model_landmarks_vec3[i] << v1;

		auto rgb_point = rgb_landmarks[0].part(i);
		// read out from obj file
		float rgb_z = 0; // = TODO;  
		//Vector3f v2(rgb_point.x(), rgb_point.y(), rgb_z); //TODO: this or the following line crashes, test and solve pls
		//rgb_landmarks_vec3[i] << v2;
	}
	return Matrix4f::Zero(); // TODO
	// return estimatePose(model_landmarks_vec3, rgb_landmarks_vec3);
}