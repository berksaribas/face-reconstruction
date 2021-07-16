#include "DenseOptimizer.h"
#include "RegularizationTerm.h"

#define SHAPE_REGULARIZATION_WEIGHT 5
#define EXPRESSION_REGULARIZATION_WEIGHT 5
#define COLOR_REGULARIZATION_WEIGHT 400

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
				new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
				new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
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

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation);
	}

	//Learn color weights only
	{
		auto vertices = get_vertices(bfm, params);
		Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
		auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
		auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, transformation_matrix, vertices);

		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, 199, 199>(
				new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(color_cost, NULL,params.col_weights.data());

		for (int i = 0; i < triangle_render.rows; i++) {
			for (int j = 0; j < triangle_render.cols; j++) {
				auto p = triangle_render.data + (i * triangle_render.cols + j) * 3;
				int triangle_id = (0 << 24) | ((int)p[2] << 16) | ((int)p[1] << 8) | ((int)p[0]);
				if (triangle_id < 56572) {
					ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBCost, 3, 199>(
						new DenseRGBCost(bfm, &image, triangle_id, j, i, &transformed_vertices)
						);
					sparse_problem.AddResidualBlock(cost_function, NULL, params.col_weights.data());
				}
			}
		}

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation);
	}

	bfm_create_obj(bfm, params);
}

void DenseOptimizer::optimize(RGBD_Image* rgbd, std::vector<dlib::full_object_detection> detected_landmarks)
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
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation);
	}

	std::cout << translation << "\n";

	//Learn position + rotation + shape weights + expression weights
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
			new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
			new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);


		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation);
	}

	//Learn color + depth
	{
		auto vertices = get_vertices(bfm, params);
		Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
		auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
		auto transformed_vertices = calculate_transformation_perspective(rgbd->image.cols, rgbd->image.rows, transformation_matrix, vertices);

		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::DENSE_QR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;
		options.max_linear_solver_iterations = 1;

		ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, 199, 199>(
			new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(color_cost, NULL, params.col_weights.data());

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, 199, 199>(
			new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, 100, 100>(
			new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
			);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());

		double model_depth_min = 9999, model_depth_max = 0;
		double image_depth_min = 9999, image_depth_max = 0;
		
		for (int j = 0; j < 68; j++) {
			double image_depth = rgbd->get_depth(detected_landmarks[0].part(j).x() / 2, detected_landmarks[0].part(j).y() / 2);
			double model_depth = transformed_vertices(bfm.landmarks[j], 2);

			if (model_depth < model_depth_min) {
				model_depth_min = model_depth;
			}
			else if (model_depth > model_depth_max) {
				model_depth_max = model_depth;
			}

			if (image_depth < image_depth_min) {
				image_depth_min = image_depth;
			}
			else if (image_depth > image_depth_max) {
				image_depth_max = image_depth;
			}
		}

		std::cout << model_depth_min << " - " << model_depth_max << "\n";
		std::cout << image_depth_min << " - " << image_depth_max << "\n";

		for (int j = 0; j < 56572; j++) {
			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBDepthCost, 4, 4, 3, 199, 100, 199>(
				new DenseRGBDepthCost(bfm, rgbd, j, &transformed_vertices, model_depth_min, model_depth_max, image_depth_min, image_depth_max)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data());
		}

		sparse_problem.SetParameterBlockConstant(rotation);
		sparse_problem.SetParameterBlockConstant(translation.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation);
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

	albedo_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false);
	triangle_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false, true);

	cv::imwrite("img_"+ std::to_string(render_number) + ".png", albedo_render);
	cv::imwrite("img_"+ std::to_string(render_number) + "_triangles.png", triangle_render);
	terminate_rendering_context();

	render_number++;
}
