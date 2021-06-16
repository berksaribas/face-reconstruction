#include <iostream>
#include <fstream>

#include "Eigen.h"
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

#include "BFM.h"

int main() {
	auto bfm = setup();
	create_random_face(bfm);
	return 0;
}