#include <iostream>
#include <fstream>

#include "Eigen.h"
//#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>

#include "FacialLandmarkDetection.h";

#define DETECT_FACIAL_LANDMARKS		1
#define SHOW_FACIAL_LANDMARKS		1

using namespace dlib;
using namespace std;

int main(int argc, char** argv) {
	//return 0;
	if (DETECT_FACIAL_LANDMARKS) {
		if (argc < 4) {
			cout << "too little arguments. Provide a valid path to the data" << endl;
			return 1;
		}
		cout << argc << endl;
		char* path = argv[3];
		cout << path << endl;
		std::vector<full_object_detection> detectedLandmarks;
		detectedLandmarks = DetectLandmarks(path, SHOW_FACIAL_LANDMARKS, true);
	}
}
