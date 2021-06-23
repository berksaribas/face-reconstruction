#pragma once
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/image_processing/render_face_detections.h>

/**
 * ICP optimizer - using Ceres for optimization.
 */
class RGBD_Image {
public:
	std::vector<dlib::full_object_detection> landmarks;
	// TODO: save the depth image

	RGBD_Image() {};

	float GetDepth(int x, int y) {
		// TODO: implement
		//landmarks[0].part(0).
		return 0.; 
	}
};