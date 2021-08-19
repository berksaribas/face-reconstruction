#include <iostream>

#include "FacialLandmarkDetection.h";
#include "BFM.h";
#include "Renderer.h"
#include "DenseOptimizer.h"
#include "RGBD_Image.h"


#define CONSTRUCT_FACE

int main(int argc, char** argv) {

#ifdef CONSTRUCT_FACE
	DenseOptimizer optimizer;
	
	cv::Mat img = cv::imread("../data/image_1.jpg");
	std::vector<dlib::full_object_detection> detectedLandmarks;
	detectedLandmarks = DetectLandmarks("../data/image_1.jpg", true, true);
	optimizer.optimize(img, detectedLandmarks);

#endif

#ifdef TRANSFER_EXP
	DenseOptimizer optimizer;
	DenseOptimizer optimizer2;

	cv::Mat img = cv::imread("../data/image_1.jpg");
	std::vector<dlib::full_object_detection> detectedLandmarks;
	detectedLandmarks = DetectLandmarks("../data/image_1.jpg", true, true);
	auto params = optimizer.optimize(img, detectedLandmarks, true);

	cv::Mat img2 = cv::imread("../data/image_2.jpg");
	std::vector<dlib::full_object_detection> detectedLandmarks2;
	detectedLandmarks2 = DetectLandmarks("../data/image_2.jpg", true, true);
	optimizer2.optimize(img2, detectedLandmarks2, false, &params.exp_weights);
#endif

	return 0;
}
