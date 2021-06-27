#pragma once

#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <Eigen.h>

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

    return bfm;
}

void bfm_create_random_face(BFM bfm) {
    int seed;
    std::cin >> seed;
    srand(seed);

    std::ofstream obj_file("export.obj");
    
    MatrixXd shape_pca_var =  Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXd shape_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXd exp_pca_var = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
    MatrixXd exp_pca_basis = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);

    VectorXd shape_weights = VectorXd::Random(199) * 0.02;
    VectorXd exp_weights = VectorXd::Random(100) * 0.02;

    MatrixXd shape_result = shape_pca_basis * (shape_pca_var * shape_weights);
    MatrixXd exp_result = exp_pca_basis * (exp_pca_var * exp_weights);
    MatrixXd result = shape_result + exp_result;

    for (int i = 0; i < 85764; i += 3) {
        obj_file << "v " << bfm.shape_mean[i] + bfm.exp_mean[i] + result(i) << " " << bfm.shape_mean[i + 1] + bfm.exp_mean[i + 1] + result(i + 1) << " " << bfm.shape_mean[i + 2] + bfm.exp_mean[i + 2] + result(i + 2) << " " << bfm.color_mean[i] << " " << bfm.color_mean[i + 1] << " " << bfm.color_mean[i + 2] << "\n";
    }

    for (int i = 0; i < 56572; i++) {
        obj_file << "f " << bfm.triangles[i] + 1 << " " << bfm.triangles[i + 56572] + 1 << " " << bfm.triangles[i + 56572 * 2] + 1 << "\n";
    }

    obj_file.close();
}