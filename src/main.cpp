#include <iostream>

#include "FacialLandmarkDetection.h";
#include "BFM.h";
//#include "CalcErrorTerm.h";
#include "Renderer.h"
#include "DenseOptimizer.h"
#include "RGBD_Image.h"

//#define DETECT_FACIAL_LANDMARKS		1
//#define SHOW_FACIAL_LANDMARKS		0

//#define BFM_CREATE_RANDOM_FACE		1
//#define RENDERER_DEMO		

#define CONSTRUCT_FACE

int main(int argc, char** argv) {
#ifdef CONSTRUCT_FACE
	DenseOptimizer optimizer;
	cv::Mat img = cv::imread("../data/tom.jpg");
	std::vector<dlib::full_object_detection> detectedLandmarks;
	detectedLandmarks = DetectLandmarks("../data/tom.jpg", false, true);
	optimizer.optimize(img, detectedLandmarks);

	//RGBD_Image *rgbd = new RGBD_Image("../data/RGBD_data/Test1/000_00_image.png","../data/RGBD_data/Test1/000_00_cloud.bin");
	//optimizer.optimize(rgbd, rgbd->landmarks);
	return 1;
#endif

#ifdef DETECT_FACIAL_LANDMARKS
	if (argc < 4) {
		std::cout << "too little arguments. Provide a valid path to the data\n";
		return 1;
	}
	std::cout << argc << "\n";
	char* path = argv[3];
	std::cout << path << "\n";
	std::vector<dlib::full_object_detection> detectedLandmarks;
	detectedLandmarks = DetectLandmarks(path, SHOW_FACIAL_LANDMARKS, true);
	std::cout << "done detecting landmarks" << std::endl;
#endif // DETECT_FACIAL_LANDMARKS 

#ifdef BFM_CREATE_RANDOM_FACE
	auto bfm_rnd = bfm_setup();
	std::cout << "finished bfm setup" << std::endl;
	Parameters params_rnd = bfm_create_random_face();
	bfm_create_obj(bfm_rnd, params_rnd);
	bfm_calc_2D_landmarks(bfm_rnd, params_rnd, 512, 512, true);
#endif // BFM_CREATE_RANDOM_FACE

#ifdef RENDERER_DEMO
	//First we load BFM data
	auto bfm = bfm_setup();
	Parameters params = bfm_mean_params();
	//Setting width and height. in real applications this would be equal to input image width/height
	//Probably we will crop the input image with the extents of landmarks
	bfm_calc_2D_landmarks(bfm, params, 512, 512, true);
	/*int width = 800;
	int height = 800;
	//We create a rendering context. Rendering context is not required if nothing is being rendered.
	auto context = init_rendering_context(width, height);
	//Creating the matrices for rotation and translation. Translating vertices with -400 on Z axis to make sure model is visible
	Eigen::Matrix3d rotation;
	rotation.setIdentity();
	Eigen::Vector3d translation = { 0, 0, -400 };
	//Creating the transformation matrix with given rotation and translation
	auto transformation_matrix = calculate_transformation_matrix(translation, rotation);
	//Transforming the vertices with the given transformation matrix and applying perspective projection
	//Here we only transform the mean shape but in real application we will have something similar to random face generator in BFM.h
	auto transformed_vertices = calculate_transformation_perspective(width, height, transformation_matrix, bfm.shape_mean);
	//After having the transformed vertices there are two use cases, following command renders the image for DENSE term
	auto rendered_result = render_mesh(context, transformed_vertices, bfm.triangles, bfm.color_mean, bfm.landmarks, true);
	cv::imwrite("img.png", rendered_result);
	//Following is for the sparse term, containing 2D landmarks:
	//This is a 68x2 matrix, each row having x and y coordinates of landmarks
	//Last parameter is for bottom left coordinate system, change it to false if you want top left.
	auto landmarks = get_transformed_landmarks(width, height, transformed_vertices, bfm.landmarks, true); 
	std::cout << landmarks(0, 0) << " " << landmarks(0, 1) << "\n";
	//Closing the rendering context
	terminate_rendering_context();>*/
#endif

#ifdef DETECT_FACIAL_LANDMARKS
	std::cout << "Calcu sparsee" << std::endl;
	if (detectedLandmarks.size() < 1) {
		std::cout << "No face could be detected" << std::endl;
		return 1;
	}
	else if (detectedLandmarks.size() > 1) {
		std::cout << "Warning: More than one face has been detected" << std::endl;
	}
	//Matrix4f Pose = GetPose(bfm, params, detectedLandmarks);
	//std::cout << Pose << std::endl;
	//CalcSparseTerm(bfm.landmarks, params, detectedLandmarks);
#endif // DETECT_FACIAL_LANDMARKS
	return 0;
}
