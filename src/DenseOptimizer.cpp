#include "DenseOptimizer.h"
#include "RegularizationTerm.h"

#define SHAPE_REGULARIZATION_WEIGHT 10
#define EXPRESSION_REGULARIZATION_WEIGHT 8
#define COLOR_REGULARIZATION_WEIGHT 0.0007

Parameters DenseOptimizer::optimize(cv::Mat image, std::vector<dlib::full_object_detection> detected_landmarks, bool skip_color, VectorXd* exp_weights)
{
	BFM bfm = bfm_setup();

	//Variables that we are going to optimize for
	double* rotation = new double[4];
	rotation[0] = 1;
	rotation[1] = 0;
	rotation[2] = 0;
	rotation[3] = 0;
	Eigen::Vector3d translation = { 0, 0, -400 };
	double* fov = new double[1];
	fov[0] = 45.0;
	Parameters params = bfm_mean_params();
	alternative_colors = new double[85764];

	//Learn position+rotation+fov (using landmarks)
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 1, SHAPE_COUNT, EXP_COUNT>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), fov, params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());
		//sparse_problem.SetParameterBlockConstant(fov);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation, fov[0]);
	}

	std::cout << translation << "\n";
	std::cout << fov[0] << "\n";

	//Learn position + rotation + shape weights + expression weights
	{
		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* shape_cost = new ceres::AutoDiffCostFunction<ShapeCostFunction, SHAPE_COUNT, SHAPE_COUNT>(
				new ShapeCostFunction(bfm, SHAPE_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(shape_cost, NULL, params.shape_weights.data());

		ceres::CostFunction* expression_cost = new ceres::AutoDiffCostFunction<ExpressionCostFunction, EXP_COUNT, EXP_COUNT>(
				new ExpressionCostFunction(bfm, EXPRESSION_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(expression_cost, NULL, params.exp_weights.data());

		for (int j = 0; j < 68; j++) {
			Vector2d detected_landmark = { detected_landmarks[0].part(j).x(), detected_landmarks[0].part(j).y() };

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 1, SHAPE_COUNT, EXP_COUNT>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], image.cols, image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), fov, params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		sparse_problem.SetParameterBlockConstant(fov);
		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation, fov[0]);
	}

	if (skip_color) {
		return params;
	}

	//Learn color weights only
	{
		auto vertices = get_vertices(bfm, params);
		Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
		auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
		auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, fov[0], transformation_matrix, vertices);

		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;

		ceres::CostFunction* color_cost = new ceres::AutoDiffCostFunction<ColorCostFunction, COLOR_COUNT, COLOR_COUNT>(
				new ColorCostFunction(bfm, COLOR_REGULARIZATION_WEIGHT)
				);
		sparse_problem.AddResidualBlock(color_cost, NULL,params.col_weights.data());

		for (int i = 0; i < triangle_render.rows; i++) {
			for (int j = 0; j < triangle_render.cols; j++) {
				auto p = triangle_render.data + (i * triangle_render.cols + j) * 3;
				int triangle_id = (0 << 24) | ((int)p[2] << 16) | ((int)p[1] << 8) | ((int)p[0]);
				if (triangle_id < 56572) {
					ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBCost, 3, COLOR_COUNT>(
						new DenseRGBCost(bfm, &image, triangle_id, j, i, &transformed_vertices)
						);
					sparse_problem.AddResidualBlock(cost_function, NULL, params.col_weights.data());
				}
			}
		}

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation, fov[0]);
	}
	
	memcpy(alternative_colors, get_colors(bfm, params), 85764 * sizeof(double));

	{
		auto vertices = get_vertices(bfm, params);
		Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
		auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
		auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, fov[0], transformation_matrix, vertices);

		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;
		for (int i = 0; i < triangle_render.rows; i++) {
			for (int j = 0; j < triangle_render.cols; j++) {
				auto p = triangle_render.data + (i * triangle_render.cols + j) * 3;
				int triangle_id = (0 << 24) | ((int)p[2] << 16) | ((int)p[1] << 8) | ((int)p[0]);
				if (triangle_id < 56572) {
					double* colors1 = alternative_colors + bfm.triangles[triangle_id + 0 * 56572] * 3;
					double* colors2 = alternative_colors + bfm.triangles[triangle_id + 1 * 56572] * 3;
					double* colors3 = alternative_colors + bfm.triangles[triangle_id + 2 * 56572] * 3;

					ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBCostAlternative, 3, 3, 3, 3>(
						new DenseRGBCostAlternative(bfm, &image, triangle_id, j, i, &transformed_vertices)
						);
					sparse_problem.AddResidualBlock(cost_function, NULL, colors1, colors2, colors3);
				}
			}
		}

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(image, bfm, params, translation, rotation, fov[0], true);
	}

	if (exp_weights != nullptr) {
		params.exp_weights = *exp_weights;
		render(image, bfm, params, translation, rotation, fov[0], true);
	}

	bfm_create_obj(bfm, params, alternative_colors);

	return params;
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
	double* fov = new double[1];
	fov[0] = 45;
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

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 1, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), fov, params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);

		// keep the shape and expression constant
		sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		sparse_problem.SetParameterBlockConstant(params.exp_weights.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation, fov[0]);
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

			ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SparseCost, 2, 4, 3, 1, 199, 100>(
				new SparseCost(bfm, detected_landmark, bfm.landmarks[j], rgbd->image.cols, rgbd->image.rows)
				);
			sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), fov, params.shape_weights.data(), params.exp_weights.data());
		}

		ceres::LocalParameterization* quaternion_parameterization = new ceres::QuaternionParameterization;
		sparse_problem.SetParameterization(rotation, quaternion_parameterization);
		sparse_problem.SetParameterBlockConstant(fov);

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation, fov[0]);
	}

	//Learn color + depth + position
	for(int iteration = 0; iteration < 10; iteration++)
	{
		auto vertices = get_vertices(bfm, params);
		Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
		auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
		auto transformed_vertices = calculate_transformation_perspective(rgbd->image.cols, rgbd->image.rows, fov[0], transformation_matrix, vertices);

		ceres::Problem sparse_problem;
		ceres::Solver::Options options;
		options.linear_solver_type = ceres::ITERATIVE_SCHUR;
		options.num_threads = 12;
		options.minimizer_progress_to_stdout = true;
		options.max_num_iterations = 1;

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

		int min_x = 9999, min_y = 9999, max_x = 0, max_y = 0;
		for (int j = 0; j < 68; j++) {
			if(detected_landmarks[0].part(j).x() < min_x) {
				min_x = detected_landmarks[0].part(j).x();
			}
			if (detected_landmarks[0].part(j).x() > max_x) {
				max_x = detected_landmarks[0].part(j).x();
			}
			if (detected_landmarks[0].part(j).y() < min_y) {
				min_y = detected_landmarks[0].part(j).y();
			}
			if (detected_landmarks[0].part(j).y() > max_y) {
				max_y = detected_landmarks[0].part(j).y();
			}
		}

		for (int i = 0; i < triangle_render.rows; i++) {
			for (int j = 0; j < triangle_render.cols; j++) {
				auto p = triangle_render.data + (i * triangle_render.cols + j) * 3;
				int triangle_id = (0 << 24) | ((int)p[2] << 16) | ((int)p[1] << 8) | ((int)p[0]);
				if (triangle_id < 56572) {
					double image_depth = rgbd->get_depth(j, i);
					if (!(i >= min_y && i <= max_y && j >= min_x && j <= max_x)) {
						continue;
					}
					if (image_depth < 0.5 || image_depth > 2) {
						continue;
					}

					ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<DenseRGBDepthCost, 6, 4, 3, 2, 199, 100, 199>(
						new DenseRGBDepthCost(bfm, rgbd, triangle_id, &model_depth_min, &model_depth_max, &image_depth_min, &image_depth_max, j, i)
						);
					sparse_problem.AddResidualBlock(cost_function, NULL, rotation, translation.data(), fov, params.shape_weights.data(), params.exp_weights.data(), params.col_weights.data());
					
					if (image_depth < image_depth_min) {
						image_depth_min = image_depth;
					}
					else if (image_depth > image_depth_max) {
						image_depth_max = image_depth;
					}

					for (int t = 0; t < 3; t++) {
						int vertex_id = bfm.triangles[triangle_id + t * 56572];
						double model_depth = transformed_vertices(vertex_id, 2);

						if (model_depth < model_depth_min) {
							model_depth_min = model_depth;
						}
						else if (model_depth > model_depth_max) {
							model_depth_max = model_depth;
						}
					}
				}
			}
		}

		std::cout << model_depth_min << " - " << model_depth_max << "\n";
		std::cout << image_depth_min << " - " << image_depth_max << "\n";
		std::cout << min_x << " - " << max_x << "\n";
		std::cout << min_y << " - " << max_y << "\n";

		sparse_problem.SetParameterBlockConstant(rotation);
		sparse_problem.SetParameterBlockConstant(translation.data());
		//sparse_problem.SetParameterBlockConstant(fov);
		//sparse_problem.SetParameterBlockConstant(params.exp_weights.data());
		//sparse_problem.SetParameterBlockConstant(params.shape_weights.data());
		//sparse_problem.SetParameterBlockConstant(params.col_weights.data());

		ceres::Solver::Summary summary;
		ceres::Solve(options, &sparse_problem, &summary);
		//std::cout << summary.BriefReport() << std::endl;

		render(rgbd->image, bfm, params, translation, rotation, fov[0]);
	}

	bfm_create_obj(bfm, params);
}

void DenseOptimizer::render(cv::Mat image, BFM bfm, Parameters params, Eigen::Vector3d translation, double* rotation, double fov, bool include_alternative){
	auto context = init_rendering_context(image.cols, image.rows);
	auto vertices = get_vertices(bfm, params);
	auto colors = get_colors(bfm, params);
	if (include_alternative) {
		for (int i = 0; i < 85764; i++) {
			colors[i] += alternative_colors[i];
			colors[i] /= 2.0;
		}
	}

	Quaterniond rotation_quat = { rotation[0], rotation[1], rotation[2], rotation[3] };
	auto transformation_matrix = calculate_transformation_matrix(translation, rotation_quat);
	auto transformed_vertices = calculate_transformation_perspective(image.cols, image.rows, fov, transformation_matrix, vertices);

	albedo_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false);
	triangle_render = render_mesh(context, image.cols, image.rows, transformed_vertices, bfm.triangles, colors, bfm.landmarks, false, true);

	cv::imwrite("img_"+ std::to_string(render_number) + ".png", albedo_render);
	//cv::imwrite("img_"+ std::to_string(render_number) + "_triangles.png", triangle_render);
	terminate_rendering_context();

	render_number++;
}
