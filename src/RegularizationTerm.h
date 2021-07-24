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
	ColorCostFunction(BFM bfm_, double weight_)
		: bfm{ bfm_ }, weight{ weight_ }
	{}

	template<typename T>
	bool operator()(T const* color_weights, T* residuals) const
	{

		for (int k = 0; k < COLOR_COUNT; k++) {
			residuals[k] = color_weights[k] * T(sqrt(weight));
		}
		
		return true;
	}

private:
	const BFM bfm;
	const double weight;
};
struct ShapeCostFunction
{
	ShapeCostFunction(BFM bfm_, double weight_)
		: bfm{ bfm_ }, weight{ weight_ }
	{}

	template<typename T>
	bool operator()(T const* shape_weights, T* residuals) const
	{

		for (int i = 0; i < SHAPE_COUNT; i++) {
			residuals[i] = shape_weights[i] * T(sqrt(weight));
		}
		return true;
	}

private:
	const BFM bfm;
	const double weight;
};
struct ExpressionCostFunction
{
	ExpressionCostFunction(BFM bfm_, double weight_)
		: bfm{ bfm_ }, weight{weight_}
	{}

	template<typename T>
	bool operator()(T const* exp_weights, T* residuals) const
	{
		for (int j = 0; j < EXP_COUNT; j++) {
			residuals[j] = exp_weights[j] * T(sqrt(weight));
		}
		return true;
	}

private:
	const BFM bfm;
	const double weight;
};