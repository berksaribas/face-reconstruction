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
	SparseCostFunction(const BFM bfm_, const dlib::point& targetPoint_, const int srcPointIdx_, const Weight& weight_)
		: m_bfm(bfm_), m_srcPointIdx(srcPointIdx_), m_targetPoint(targetPoint_), m_weight(weight_)
	{
	}

	template<typename T>
	bool operator()(const T* shape_weights, const T* exp_weights, const T* col_weights, T* residual) const // const T* Transform, 
	{
		//auto cosinus = cos(angle[0]);
		//auto sinus = sin(angle[0]);
		//Point2D p_transformed = point1;
		//T trans_x = cosinus * point1.x - sinus * point1.y + tx[0];
		//T trans_y = sinus * point1.x + cosinus * point1.y + ty[0];
		Parameters params;
		params.shape_weights = shape_weights;
		params.exp_weights = exp_weights;
		params.col_weights = col_weights;
		MatrixXf landmarks = bfm_calc_2D_landmarks(m_bfm, params);
		auto point2_eigen = landmarks.rows()[m_srcPointIdx];
		std::cout << "Src point eigen: " + point2_eigen << std::endl;
		Point2D point2(0, 0); // TODO!
		auto res1 = m_weight.w * (m_targetPoint.x - point2.x);
		auto res2 = m_weight.w * (m_targetPoint.y - point2.y);
		//auto res2 = weight.w * (targetPoint.z - point2.z);
		residual[0] = T(res1);
		residual[1] = T(res2);
		//residual[1] = T(res2);
		return true;
	}

	static ceres::CostFunction* create(BFM bfm, const dlib::point& targetPoint, const int srcPointIdx, const Weight& weight) {
		return new ceres::AutoDiffCostFunction<SparseCostFunction, 2, 199, 100, 199>(
			new SparseCostFunction(bfm, targetPoint, srcPointIdx, weight)
			);
	}

private:
	const BFM m_bfm;
	const dlib::point m_targetPoint;
	const int m_srcPointIdx;
	const Weight m_weight;
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