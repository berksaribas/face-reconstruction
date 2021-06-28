#pragma once
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <Eigen.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

GLFWwindow* init_rendering_context(int width, int height);
cv::Mat render_mesh(GLFWwindow* window, MatrixXd vertices, int* triangles, double* colors, std::vector<int> landmarks, bool draw_landmarks);
void terminate_rendering_context();
Matrix4d calculate_transformation_matrix(Eigen::Vector3d translation, Eigen::Matrix3d rotation);
Matrix4d calculate_perspective_matrix(double angle, double aspect_ratio, double near, double far);
MatrixXd calculate_transformation_perspective(int width, int height, Matrix4d transformation_matrix, double* vertices);
MatrixXd get_transformed_landmarks(int width, int height, MatrixXd vertices, std::vector<int> landmarks, bool bottom_left);