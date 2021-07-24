#pragma once
#define GLFW_INCLUDE_GLU
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <Eigen.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <ceres/ceres.h>

GLFWwindow* init_rendering_context(int width, int height);
cv::Mat render_mesh(GLFWwindow* window, int width, int height, MatrixXd vertices, int* triangles, double* colors, std::vector<int> landmarks, bool draw_landmarks, bool render_triangle_id = false);
void terminate_rendering_context();
MatrixXd get_transformed_landmarks(int width, int height, MatrixXd vertices, std::vector<int> landmarks, bool bottom_left);

template <typename T>
Matrix<T, 4, 4> calculate_perspective_matrix(T angle, T aspect, T zNear, T zFar) {
    T const rad = angle * T(M_PI / 180);
    T const tanHalfFovy = tan(rad / static_cast<T>(2));

    Matrix<T, 4, 4> projection_matrix;
    projection_matrix.setConstant(T(0));

    projection_matrix(0, 0) = static_cast<T>(1) / (aspect * tanHalfFovy);
    projection_matrix(1, 1) = static_cast<T>(1) / (tanHalfFovy);
    projection_matrix(2, 2) = -(zFar + zNear) / (zFar - zNear);
    projection_matrix(2, 3) = -static_cast<T>(1);
    projection_matrix(3, 2) = -(static_cast<T>(2)* zFar* zNear) / (zFar - zNear);
    projection_matrix.transposeInPlace();

    return projection_matrix;
}

template <typename T>
static Matrix<T, 4, 4> calculate_transformation_matrix(Matrix<T, 3, 1> translation, Matrix<T, 3, 3> rotation) {
    Matrix<T, 4, 4> transformation;
    transformation.setIdentity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}

template <typename T>
static Matrix<T, 4, 4> calculate_transformation_matrix(Matrix<T, 3, 1> translation, Quaternion<T> rotation) {
    return calculate_transformation_matrix<T>(translation, rotation.toRotationMatrix());
}

template <typename T>
static Matrix<T, -1, -1> calculate_transformation_perspective(double width, double height, T fov, Matrix<T, 4, 4> transformation_matrix, Matrix<T, -1, -1> mapped_vertices) {
    //Transposing Vertices and adding an extra column for W
    mapped_vertices.conservativeResize(mapped_vertices.rows(), mapped_vertices.cols() + 1);
    mapped_vertices.col(3).fill(T(1));
    mapped_vertices.transposeInPlace();

    //Transformation Matrix
    Matrix<T, -1, -1> result = transformation_matrix * mapped_vertices;

    //Projecting
    T ar = T(width / height);
    T n = T(0.1);
    T f = T(10000.0);
    Matrix<T, 4, 4> projection_matrix = calculate_perspective_matrix(fov, ar, n, f);
    result = projection_matrix * result;

    result.transposeInPlace();
    return result;
}

static MatrixXd calculate_transformation_perspective(int width, int height, double fov, Matrix4d transformation_matrix, double* vertices) {
    MatrixXd mapped_vertices = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(vertices, 28588, 3);
    return calculate_transformation_perspective<double>((double)width, (double)height, fov, transformation_matrix, mapped_vertices);
}