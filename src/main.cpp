#include <iostream>
#include <fstream>

#include "Eigen.h"
//#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>

#include "FacialLandmarkDetection.h";

#define DETECT_FACIAL_LANDMARKS		1
#define SHOW_FACIAL_LANDMARKS		1
#define sourcePath					"C:/Users/Philip/Desktop/Github/Uni/MS2-Git/3D_Scanning/face-reconstruction/data/Face3/realsense_face3_sf2_deep.png"

using namespace dlib;
using namespace std;

int main(int argc, char** argv) {
	//return 0;
	if (DETECT_FACIAL_LANDMARKS) {
		std::vector<full_object_detection> detectedLandmarks;
		detectedLandmarks = DetectLandmarks(sourcePath, SHOW_FACIAL_LANDMARKS, true);
	}
}
