#pragma once

// The Google logging library (GLOG), used in Ceres, has a conflict with Windows defined constants. This definitions prevents GLOG to use the same constants
#define GLOG_NO_ABBREVIATED_SEVERITIES

#include <ceres/ceres.h>

/**
 * ICP optimizer - using Ceres for optimization.
 */
/*class CeresICPOptimizer { //: public ICPOptimizer {
public:

	CeresICPOptimizer() :
		m_nIterations{ 20 } {}

	virtual Eigen::Matrix4f estimatePose(const PointCloud& source, const PointCloud& target, Eigen::Matrix4f initialPose = Matrix4f::Identity()) override {
		// Build the index of the FLANN tree (for fast nearest neighbor lookup).
		m_nearestNeighborSearch->buildIndex(target.getPoints());

		// The initial estimate can be given as an argument.
		Matrix4f estimatedPose = initialPose;

		// We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
		// the rotation angle) and 3 parameters for the translation vector. 
		double incrementArray[6];
		auto poseIncrement = PoseIncrement<double>(incrementArray);
		poseIncrement.setZero();

		for (int i = 0; i < m_nIterations; ++i) {
			// Compute the matches.
			std::cout << "Matching points ..." << std::endl;
			clock_t begin = clock();

			auto transformedPoints = transformPoints(source.getPoints(), estimatedPose);
			auto transformedNormals = transformNormals(source.getNormals(), estimatedPose);

			auto matches = m_nearestNeighborSearch->queryMatches(transformedPoints);
			pruneCorrespondences(transformedNormals, target.getNormals(), matches);

			clock_t end = clock();
			double elapsedSecs = double(end - begin) / CLOCKS_PER_SEC;
			std::cout << "Completed in " << elapsedSecs << " seconds." << std::endl;

			// Prepare point-to-point and point-to-plane constraints.
			ceres::Problem problem;
			prepareConstraints(transformedPoints, target.getPoints(), target.getNormals(), matches, poseIncrement, problem);

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);

			// Run the solver (for one iteration).
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			//std::cout << summary.FullReport() << std::endl;

			// Update the current pose estimate (we always update the pose from the left, using left-increment notation).
			Eigen::Matrix4f matrix = PoseIncrement<double>::convertToMatrix(poseIncrement);
			estimatedPose = PoseIncrement<double>::convertToMatrix(poseIncrement) * estimatedPose;
			poseIncrement.setZero();

			std::cout << "Optimization iteration done." << std::endl;
		}

		return estimatedPose;
	}


private:
	void configureSolver(ceres::Solver::Options& options) {
		// Ceres options.
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = 1;
		options.max_num_iterations = 1;
		options.num_threads = 8;
	}

	void prepareConstraints(const std::vector<Eigen::Vector3f>& sourcePoints, const std::vector<Eigen::Vector3f>& targetPoints, const std::vector<Vector3f>& targetNormals, const std::vector<Match> matches, const PoseIncrement<double>& poseIncrement, ceres::Problem& problem) const {
		const unsigned nPoints = sourcePoints.size();

		for (unsigned i = 0; i < nPoints; ++i) {
			const auto match = matches[i];
			if (match.idx >= 0) {
				const auto& sourcePoint = sourcePoints[i];
				const auto& targetPoint = targetPoints[match.idx];
				const auto& weight = match.weight;

				if (!sourcePoint.allFinite() || !targetPoint.allFinite())
					continue;

				// TODO: Create a new point-to-point cost function and add it as constraint (i.e. residual block) 
				// to the Ceres problem.
				ceres::CostFunction* costPoint = PointToPointConstraint::create(sourcePoint, targetPoint, weight);
				//ceres::CostFunction* costFun = new ceres::AutoDiffCostFunction<PointToPointConstraint, 3, 6>();
				problem.AddResidualBlock(
					costPoint,
					nullptr, poseIncrement.getData()
				);

				if (m_bUsePointToPlaneConstraints) {
					const auto& targetNormal = targetNormals[match.idx];

					if (!targetNormal.allFinite())
						continue;

					// TODO: Create a new point-to-plane cost function and add it as constraint (i.e. residual block) 
					// to the Ceres problem.
					ceres::CostFunction* costPlane = PointToPlaneConstraint::create(sourcePoint, targetPoint, targetNormal, weight);
					//ceres::CostFunction* costFunPlane = new ceres::AutoDiffCostFunction<PointToPlaneConstraint, 1, 6>(costPlane);

					problem.AddResidualBlock(
						costPlane,
						nullptr, poseIncrement.getData()
					);
				}
			}
		}
	}

	private:
		unsigned m_nIterations;
};*/