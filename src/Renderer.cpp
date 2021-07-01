#include "Renderer.h"
#include <iostream>

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

Matrix4d calculate_perspective_matrix(double angle, double aspect_ratio, double near, double far) {
    double scale = tan(angle * 0.5 * M_PI / 180) * near;

    double r = aspect_ratio * scale;
    double l = -r;
    double t = scale;
    double b = -t;

    Matrix4d projection_matrix;
    projection_matrix << 2 * near / (r - l), 0, (r + l) / (r - l), 0,
        0, 2 * near / (t - b), (t + b) / (t - b), 0,
        0, 0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
        0, 0, -1, 0;

    return projection_matrix;
}

MatrixXd get_transformed_landmarks(int width, int height, MatrixXd vertices, std::vector<int> landmarks, bool bottom_left) {
    MatrixXd transformed_landmarks(landmarks.size(), 2);
    for (int i = 0; i < landmarks.size(); i++) {
        int vertex_index = landmarks[i];
        double w = vertices(vertex_index, 3);

        transformed_landmarks(i, 0) = (vertices(vertex_index, 0) / w + 1) / 2 * width;
        transformed_landmarks(i, 1) = (vertices(vertex_index, 1) / w + 1) / 2 * height;

        if (!bottom_left) {
            transformed_landmarks(i, 1) = height - (vertices(vertex_index, 1) / w + 1) / 2 * height;
        }
    }

    return transformed_landmarks;
}

cv::Mat render_mesh(GLFWwindow* window, MatrixXd vertices, int* triangles, double* colors, std::vector<int> landmarks, bool draw_landmarks) {
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 56572; i++) {
        for (int t = 0; t < 3; t++) {
            int vertex_index = triangles[i + t * 56572];

            double w = vertices(vertex_index, 3);

            glColor3d(colors[vertex_index * 3], colors[vertex_index * 3 + 1], colors[vertex_index * 3 + 2]);
            glVertex3d(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
        }
    }
    glEnd();

    if (draw_landmarks) {
        glPointSize(4);
        glBegin(GL_POINTS);
        for (int i = 0; i < landmarks.size(); i++) {
            int vertex_index = landmarks[i];
            double w = vertices(vertex_index, 3);
            glColor3d(1, 0, 0);
            glVertex3d(vertices(vertex_index, 0) / w, vertices(vertex_index, 1) / w, vertices(vertex_index, 2) / w);
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