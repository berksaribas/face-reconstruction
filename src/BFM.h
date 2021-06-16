#pragma once

#include <string>
#include "H5Cpp.h"
#include <iostream>
#include <algorithm>
#include <fstream>

std::string triangles_path = "../data/shape representer cells.bin"; // 3x56572
std::string texture_path = "../data/color model mean.bin"; // 85764

std::string shape_mean_path = "../data/shape model mean.bin"; // 85764
std::string shape_pca_variance_path = "../data/shape model pcaVariance.bin"; // 199
std::string shape_pca_basis_path = "../data/shape model pcaBasis.bin"; // 85764x199

std::string expression_mean_path = "../data/expression model mean.bin"; // 85764
std::string expression_pca_variance_path = "../data/expression model pcaVariance.bin"; // 100
std::string expression_pca_basis_path = "../data/expression model pcaBasis.bin"; // 85764x100

std::string bfm_landmarks_path = "../data/Landmarks68_model2017-1_face12_nomouth.anl";

struct BFM {
    int* triangles;
    float* mean_tex;

    float* shape_mean;
    float* shape_pca_var;
    float* shape_pca_basis;

    float* exp_mean;
    float* exp_pca_var;
    float* exp_pca_basis;

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

BFM setup() {
    BFM bfm;
    bfm.triangles = (int*)load_binary_data(triangles_path.c_str());

    bfm.mean_tex = (float*)load_binary_data(texture_path.c_str());

    bfm.shape_mean = (float*)load_binary_data(shape_mean_path.c_str());
    bfm.shape_pca_var = (float*)load_binary_data(shape_pca_variance_path.c_str());
    bfm.shape_pca_basis = (float*)load_binary_data(shape_pca_basis_path.c_str());

    bfm.exp_mean = (float*)load_binary_data(expression_mean_path.c_str());
    bfm.exp_pca_var = (float*)load_binary_data(expression_pca_variance_path.c_str());
    bfm.exp_pca_basis = (float*)load_binary_data(expression_pca_basis_path.c_str());

    load_landmarks(bfm);

    return bfm;
}

void create_random_face(BFM bfm) {
    int seed;
    std::cin >> seed;
    srand(seed);

    std::ofstream obj_file("export.obj");
    
    MatrixXf shape_pca_var =  Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_var, 199, 1);
    MatrixXf shape_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 199);

    MatrixXf exp_pca_var = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 100, 1);
    MatrixXf exp_pca_basis = Map<Matrix<float, Dynamic, Dynamic, RowMajor>>(bfm.shape_pca_basis, 85764, 100);

    VectorXf shape_weights = VectorXf::Random(199) * 0.02;
    VectorXf exp_weights = VectorXf::Random(100) * 0.02;


    MatrixXf shape_result = shape_pca_basis * (shape_pca_var * shape_weights);
    MatrixXf exp_result = exp_pca_basis * (exp_pca_var * exp_weights);
    MatrixXf result = shape_result + exp_result;

    for (int i = 0; i < 85764; i += 3) {
        obj_file << "v " << bfm.shape_mean[i] + result(i) << " " << bfm.shape_mean[i + 1] + result(i + 1) << " " << bfm.shape_mean[i + 2] + result(i + 2) << " " << bfm.mean_tex[i] << " " << bfm.mean_tex[i + 1] << " " << bfm.mean_tex[i + 2] << "\n";
    }

    for (int i = 0; i < 56572; i++) {
        obj_file << "f " << bfm.triangles[i] + 1 << " " << bfm.triangles[i + 56572] + 1 << " " << bfm.triangles[i + 56572 * 2] + 1 << "\n";
    }

    obj_file.close();
}