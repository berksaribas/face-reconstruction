#pragma once
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#include <Eigen.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

GLFWwindow* init_rendering_context(int width, int height);
cv::Mat render_mesh(GLFWwindow* window, MatrixXf vertices, int* triangles, float* colors, std::vector<int> landmarks, bool draw_landmarks);
void terminate_rendering_context();
Matrix4f calculate_transformation_matrix(Eigen::Vector3f translation, Eigen::Matrix3f rotation);
Matrix4f calculate_perspective_matrix(float angle, float aspect_ratio, float near, float far);
MatrixXf calculate_transformation_perspective(int width, int height, Matrix4f transformation_matrix, float* vertices);
MatrixXf get_transformed_landmarks(int width, int height, MatrixXf vertices, std::vector<int> landmarks, bool bottom_left);