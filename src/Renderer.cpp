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

    glewInit();

    return window;
}

void terminate_rendering_context() {
    glfwTerminate();
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

cv::Mat render_mesh(GLFWwindow* window, int width, int height, MatrixXd vertices, int* triangles, double* colors, std::vector<int> landmarks, bool draw_landmarks, bool render_triangle_id) {
    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    GLuint depthrenderbuffer;
    glGenRenderbuffers(1, &depthrenderbuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, DrawBuffers);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, width, height);

    if (render_triangle_id) {
        glShadeModel(GL_FLAT);
    }
    else {
        glShadeModel(GL_SMOOTH);
    }

    glBegin(GL_TRIANGLES);
    for (int i = 0; i < 56572; i++) {
        unsigned char* p = (unsigned char*)&i;

        for (int t = 0; t < 3; t++) {
            int vertex_index = triangles[i + t * 56572];
            double w = vertices(vertex_index, 3);

            if (render_triangle_id) {
                glColor3ub(p[2], p[1], p[0]);
            }
            else {
                glColor3d(colors[vertex_index * 3], colors[vertex_index * 3 + 1], colors[vertex_index * 3 + 2]);
            }

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
    cv::Mat img(height, width, CV_8UC3, gl_texture_bytes);
    cv::flip(img, img, 0);

    glDeleteFramebuffers(1, &fbo);

    glfwSwapBuffers(window);
    glfwPollEvents();
    
    return img;
}