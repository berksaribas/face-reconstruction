#include <iostream>

#include "FacialLandmarkDetection.h";
#include "BFM.h";

#define DETECT_FACIAL_LANDMARKS		1
#define SHOW_FACIAL_LANDMARKS		1

#define BFM_CREATE_RANDOM_FACE		1

int main(int argc, char** argv) {
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
#endif // DETECT_FACIAL_LANDMARKS

#ifdef BFM_CREATE_RANDOM_FACE
	auto bfm = bfm_setup();
	bfm_create_random_face(bfm);
#endif // BFM_CREATE_RANDOM_FACE

	return 0;
}
