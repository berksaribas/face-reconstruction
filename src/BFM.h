#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <Eigen.h>
#include <renderer.h>

std::string triangles_path = "../data/shape representer cells.bin"; // 3x56572

std::string color_mean_path = "../data/color model mean.bin"; // 85764
std::string color_pca_variance_path = "../data/color model pcaVariance.bin"; // 199
std::string color_pca_basis_path = "../data/color model pcaBasis.bin"; // 85764x199

std::string shape_mean_path = "../data/shape model mean.bin"; // 85764
std::string shape_pca_variance_path = "../data/shape model pcaVariance.bin"; // 199
std::string shape_pca_basis_path = "../data/shape model pcaBasis.bin"; // 85764x199

std::string expression_mean_path = "../data/expression model mean.bin"; // 85764
std::string expression_pca_variance_path = "../data/expression model pcaVariance.bin"; // 100
std::string expression_pca_basis_path = "../data/expression model pcaBasis.bin"; // 85764x100

std::string bfm_landmarks_path = "../data/Landmarks68_model2017-1_face12_nomouth.anl";

struct BFM {
    int* triangles;

    double* color_mean;
    double* color_pca_var;
    double* color_pca_basis;

    double* shape_mean;
    double* shape_pca_var;
    double* shape_pca_basis;

    double* exp_mean;
    double* exp_pca_var;
    double* exp_pca_basis;

    std::vector<int> landmarks;

    float* vertices;
};

struct Parameters {
    VectorXd shape_weights;
    VectorXd exp_weights;
    VectorXd col_weights;
};

void load_landmarks(BFM& bfm) {
    std::ifstream infile(bfm_landmarks_path);
    int a;
    while (infile >> a)
    {
        bfm.landmarks.push_back(a);
    }
    infile.close();
}

void* load_binary_data(const char* filename) {
    FILE* in_file = fopen(filename, "rb");
    if (!in_file) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }

    struct stat sb;
    if (stat(filename, &sb) == -1) {
        perror("stat");
        exit(EXIT_FAILURE);
    }

    void* file_contents = (char*)malloc(sb.st_size);
    fread(file_contents, sb.st_size, 1, in_file);

    fclose(in_file);

    return file_contents;
}

double* convert_to_double(float* data, int size) {
    double* double_data = new double[size];
    for (int i = 0; i < size; i++) {
        double_data[i] = (double) data[i];
    }
    delete[] data;

    return double_data;
}

BFM bfm_setup() {
    BFM bfm;
    bfm.triangles = (int*)load_binary_data(triangles_path.c_str());

    bfm.color_mean = convert_to_double((float*)load_binary_data(color_mean_path.c_str()), 85764);
    bfm.color_pca_var = convert_to_double((float*)load_binary_data(color_pca_variance_path.c_str()), 199);
    bfm.color_pca_basis = convert_to_double((float*)load_binary_data(color_pca_basis_path.c_str()), 85764*199);

    bfm.shape_mean = convert_to_double((float*)load_binary_data(shape_mean_path.c_str()), 85764);
    bfm.shape_pca_var = convert_to_double((float*)load_binary_data(shape_pca_variance_path.c_str()), 199);
    bfm.shape_pca_basis = convert_to_double((float*)load_binary_data(shape_pca_basis_path.c_str()), 85764 * 199);

    bfm.exp_mean = convert_to_double((float*)load_binary_data(expression_mean_path.c_str()), 85764);
    bfm.exp_pca_var = convert_to_double((float*)load_binary_data(expression_pca_variance_path.c_str()), 100);
    bfm.exp_pca_basis = convert_to_double((float*)load_binary_data(expression_pca_basis_path.c_str()), 85764 * 100);

    load_landmarks(bfm);

    double* vertices = new double[85764];

    return bfm;
}

Parameters bfm_mean_params() {
    Parameters params;
    params.shape_weights = VectorXd::Zero(199) * 0.02;
    params.exp_weights = VectorXd::Zero(100) * 0.02;
    params.col_weights = VectorXd::Zero(199) * 0.02;
    return params;
}

Parameters bfm_create_random_face() {
    int seed = 5;
    //std::cout << "Enter a seed: ";
    //std::cin >> seed;
    srand(seed);

    Parameters params;
    params.shape_weights = VectorXd::Random(199) * 0.02;
    params.exp_weights = VectorXd::Random(100) * 0.02;
    params.col_weights = VectorXd::Random(199) * 0.02;
    return params;
}

void bfm_create_obj(BFM bfm, Parameters params) {
    std::ofstream obj_file("export.obj");

    /*MatrixXf shape_pca_var = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXf shape_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXf exp_pca_var = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
    MatrixXf exp_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);

    MatrixXf shape_result = shape_pca_basis * (shape_pca_var * params.shape_weights);
    MatrixXf exp_result = exp_pca_basis * (exp_pca_var * params.exp_weights);
    MatrixXf result = shape_result + exp_result;*/
    
    MatrixXd shape_pca_var =  Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXd shape_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXd exp_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
    MatrixXd exp_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);

    MatrixXd shape_result = shape_pca_basis * (shape_pca_var * params.shape_weights);
    MatrixXd exp_result = exp_pca_basis * (exp_pca_var * params.exp_weights);
    MatrixXd result = shape_result + exp_result;

    for (int i = 0; i < 85764; i += 3) {
        obj_file << "v " << bfm.shape_mean[i] + bfm.exp_mean[i] + result(i) << " " << bfm.shape_mean[i + 1] + bfm.exp_mean[i + 1] + result(i + 1) << " " << bfm.shape_mean[i + 2] + bfm.exp_mean[i + 2] + result(i + 2) << " " << bfm.color_mean[i] << " " << bfm.color_mean[i + 1] << " " << bfm.color_mean[i + 2] << "\n";
    }

    for (int i = 0; i < 56572; i++) {
        obj_file << "f " << bfm.triangles[i] + 1 << " " << bfm.triangles[i + 56572] + 1 << " " << bfm.triangles[i + 56572 * 2] + 1 << "\n";
    }

    obj_file.close();
}

double* get_vertices(BFM bfm, Parameters params) {
    MatrixXd shape_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXd shape_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXd exp_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
    MatrixXd exp_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);

    MatrixXd shape_result = shape_pca_basis * (shape_pca_var * params.shape_weights);
    MatrixXd exp_result = exp_pca_basis * (exp_pca_var * params.exp_weights);
    MatrixXd result = shape_result + exp_result;
    MatrixXd mapped_vertices = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_mean, 85764, 1);

    double* sum = new double[85764];
    for (int i = 0; i < 85764; i ++) {
        sum[i] = bfm.shape_mean[i] + bfm.exp_mean[i] + result(i);
    }
    return sum;
}

MatrixXd bfm_calc_2D_landmarks(BFM bfm, Parameters params, int width=800, int height=800, bool createImg=false) {
    //We create a rendering context. Rendering context is not required if nothing is being rendered.
    auto context = init_rendering_context(width, height);
    //Creating the matrices for rotation and translation. Translating vertices with -400 on Z axis to make sure model is visible
    Eigen::Matrix3d rotation;
    rotation.setIdentity();
    Eigen::Vector3d translation = { 0, 0, -400 }; // TODO: can we generalize 400 to width / 2?
    //Creating the transformation matrix with given rotation and translation
    auto transformation_matrix = calculate_transformation_matrix(translation, rotation);
    //Transforming the vertices with the given transformation matrix and applying perspective projection
    //Here we only transform the mean shape but in real application we will have something similar to random face generator in BFM.h
    //auto transformed_vertices = calculate_transformation_perspective(width, height, transformation_matrix, bfm.shape_mean);
    auto transformed_vertices = calculate_transformation_perspective(width, height, transformation_matrix, get_vertices(bfm, params));
    //After having the transformed vertices there are two use cases, following command renders the image for DENSE term
    auto rendered_result = render_mesh(context, transformed_vertices, bfm.triangles, bfm.color_mean, bfm.landmarks, true);
    if (createImg) {
        cv::imwrite("img.png", rendered_result);
    }
    //Following is for the sparse term, containing 2D landmarks:
    //This is a 68x2 matrix, each row having x and y coordinates of landmarks
    //Last parameter is for bottom left coordinate system, change it to false if you want top left.
    auto landmarks = get_transformed_landmarks(width, height, transformed_vertices, bfm.landmarks, true); 
    //std::cout << landmarks(0, 0) << " " << landmarks(0, 1) << "\n";
    return landmarks;
    //return MatrixXf::Zero();
}