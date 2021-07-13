#include "DenseOptimizer.h"
#include "RegularizationTerm.h"

void DenseOptimizer::optimize(cv::Mat image, std::vector<dlib::full_object_detection> detected_landmarks)
{
	BFM bfm = bfm_setup();

	//Variables that we are going to optimize for
	double* rotation = new double[4];
	rotation[0] = 1;
	rotation[1] = 0;
	rotation[2] = 0;
	rotation[3] = 0;
	Eigen::Vector3d translation = { 0, 0, -400 };
	Parameters params = bfm_mean_params();

	//Learn position+rotation (using landmarks)
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		for (int j = 0; j < 199; j++) {
			sparse_problem.SetParameterUpperBound(params.shape_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.shape_weights.data(), j, -1);
		}

		for (int j = 0; j < 100; j++) {
			sparse_problem.SetParameterUpperBound(params.exp_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.exp_weights.data(), j, -1);
		}

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation);
	}

	//Learn position + rotation + shape weights + expression weights
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
				new ShapeCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
				new ExpressionCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());



		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		for (int j = 0; j < 199; j++) {
			sparse_problem.SetParameterUpperBound(params.shape_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.shape_weights.data(), j, -1);
		}

		for (int j = 0; j < 100; j++) {
			sparse_problem.SetParameterUpperBound(params.exp_weights.data(), j, 1);
			sparse_problem.SetParameterLowerBound(params.exp_weights.data(), j, -1);
		}
		

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation);
	}

	//Learn color weights only
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, 199, 199>(
				new ColorCostFunction(bfm)
				);
		sparse_problem.AddResidualBlock(color_cost, NULL,params.col_weights.data());


		for (int j = 0; j < 28588; j++) {
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBCost, 3, 4, 3, 199, 100, 199>(
				new DenseRGBCost(bfm, &image, j)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data());
		}
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());
		sparse_problem.SetParameterBlockConstant(rotation);
		sparse_problem.SetParameterBlockConstant(translation.data());
		//for (int j = 0; j < 199; j++) {
		//	sparse_problem.SetParameterUpperBound(params.col_weights.data(), j, 3);
		//	sparse_problem.SetParameterLowerBound(params.col_weights.data(), j, -3);
		//}

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation);
	}

	bfm_create_obj(bfm, params);
}

void DenseOptimizer::render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation){
	auto context = init_rendering_context(image.cols, image.rows);
	auto vertices = get_vertices(bfm, params);
	auto colors = get_colors(bfm, params);
	Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
	auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
	auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, transformation_matrix, vertices);
	auto rendered = render_mesh(context, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false);

	cv::imwrite("img_"+ std::to_string(render_number) + ".png", rendered);
	terminate_rendering_context();

	render_number++;
}
