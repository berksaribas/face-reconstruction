#include "Renderer.h"
#include <iostream>

float deg2rad(float degrees) {
    return degrees * 4.0 * atan(1.0) / 180.0;
}

GLFWwindow* init_rendering_context(int width, int height) {
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return nullptr;

    /* Create a windowed mode window and its OpenGL context */
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    window = glfwCreateWindow(width, height, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return nullptr;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glShadeModel(GL_SMOOTH);

    return window;
}

void terminate_rendering_context() {
    glfwTerminate();
}

Matrix4f calculate_transformation_matrix(Eigen::Vector3f translation, Eigen::Matrix3f rotation) {
    Eigen::Matrix4f transformation;
    transformation.setIdentity();
    transformation.block<3, 3>(0, 0) = rotation;
    transformation.block<3, 1>(0, 3) = translation;

    return transformation;
}

Matrix4f calculate_perspective_matrix(float angle, float aspect_ratio, float near, float far) {
    float scale = tan(angle * 0.5 * M_PI / 180) * near;

    float r = aspect_ratio * scale;
    float l = -r;
    float t = scale;
    float b = -t;

    Matrix4f projection_matrix;
    projection_matrix << 2 * near / (r - l), 0, (r + l) / (r - l), 0,
        0, 2 * near / (t - b), (t + b) / (t - b), 0,
        0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
        0, 0, -1, 0;

    return projection_matrix;
}

MatrixXf calculate_transformation_perspective(int width, int height, Matrix4f transformation_matrix, float* vertices) {
    //Transposing Vertices and adding an extra column for W
    MatrixXf mapped_vertices = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(vertices, 28588, 3);
    mapped_vertices.conservativeResize(mapped_vertices.rows(), mapped_vertices.cols() + 1);
    mapped_vertices.col(3).fill(1);
    mapped_vertices.transposeInPlace();

    //Transformation Matrix
    MatrixXf result = transformation_matrix * mapped_vertices;

    //Projecting
    float ar = (float)width / (float)height;
    float n = 0.1;
    float f = -1;
    Matrix4f projection_matrix = calculate_perspective_matrix(45, ar, n, f);
    result = projection_matrix * result;
    
    result.transposeInPlace();
    return result;
}

MatrixXf get_transformed_landmarks(int width, int height, MatrixXf vertices, std::vector<int> landmarks, bool bottom_left) {
    MatrixXf transformed_landmarks(landmarks.size(), 2);
    for (int i = 0; i < landmarks.size(); i++) {
        int vertex_index = landmarks[i];
        float w = vertices(vertex_index, 3);

        transformed_landmarks(i, 0) = (vertices(vertex_index, 0) / w + 1) / 2 * width;
        transformed_landmarks(i, 1) = (vertices(vertex_index, 1) / w + 1) / 2 * height;

        if (!bottom_left) {
            transformed_landmarks(i, 1) = height - (vertices(vertex_index, 1) / w + 1) / 2 * height;
        }
    }

    return transformed_landmarks;
}

cv::Mat render_mesh(GLFWwindow* window, MatrixXf vertices, int* triangles, float* colors, std::vector<int> landmarks, bool draw_landmarks) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 56572; i++) {
        for (int t = 0; t < 3; t++) {
            int vertex_index = triangles[i + t * 56572];

            float w = vertices(vertex_index, 3);

            glColor3f(colors[vertex_index * 3], colors[vertex_index * 3 + 1], colors[vertex_index * 3 + 2]);
            glVertex3f(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
        }
    }
    glEnd();

    if (draw_landmarks) {
        glPointSize(4);
        glBegin(GL_POINTS);
        for (int i = 0; i < landmarks.size(); i++) {
            int vertex_index = landmarks[i];
            float w = vertices(vertex_index, 3);
            glColor3f(1, 0, 0);
            glVertex3f(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
        }
        glEnd();
    }
        

    unsigned char* gl_texture_bytes = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
    glReadPixels(0, 0, width, height, 0x80E0, GL_UNSIGNED_BYTE, gl_texture_bytes);
    cv::Mat img(width, height, CV_8UC3, gl_texture_bytes);
    cv::flip(img, img, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
    
    return img;
}