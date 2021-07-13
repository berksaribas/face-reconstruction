#pragma once
#include <vector>
#include "utils/points.h"
#include "ceres/ceres.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Eigen.h"
#include "BFM.h"

struct ColorCostFunction
{
	ColorCostFunction(BFM bfm_)
		: bfm{ bfm_ }
	{}

	template<typename T>
	bool operator()(T const* color_weights, T* residuals) const
	{

		for (int k = 0; k < 199; k++) {
			residuals[k] = T(color_weights[k]/ T(sqrt(bfm.color_pca_var[k])));
		}
		
		return true;
	}

private:
	const BFM bfm;
};
struct ShapeCostFunction
{
	ShapeCostFunction(BFM bfm_)
		: bfm{ bfm_ }
	{}

	template<typename T>
	bool operator()(T const* shape_weights, T* residuals) const
	{

		for (int i = 0; i < 199; i++) {
			residuals[i] = T(shape_weights[i]/ T(sqrt(bfm.shape_pca_var[i])));
		}
		return true;
	}

private:
	const BFM bfm;
};
struct ExpressionCostFunction
{
	ExpressionCostFunction(BFM bfm_)
		: bfm{ bfm_ }
	{}

	template<typename T>
	bool operator()(T const* exp_weights, T* residuals) const
	{
		for (int j = 0; j < 100; j++) {
			residuals[j] = T(exp_weights[j]/T(sqrt(bfm.exp_pca_var[j])));
		}
		return true;
	}

private:
	const BFM bfm;
};