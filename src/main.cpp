#include <iostream>

#include "FacialLandmarkDetection.h";
#include "BFM.h";
#include "CalcErrorTerm.h";

#define DETECT_FACIAL_LANDMARKS		1
#define SHOW_FACIAL_LANDMARKS		0

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
	std::cout << "done detecting landmarks" << std::endl;
#endif // DETECT_FACIAL_LANDMARKS

#ifdef BFM_CREATE_RANDOM_FACE
	auto bfm = bfm_setup();
	std::cout << "finished bfm setup" << std::endl;
	Parameters params = bfm_create_random_face(bfm);
	bfm_create_obj(bfm, params);
#endif // BFM_CREATE_RANDOM_FACE

#ifdef DETECT_FACIAL_LANDMARKS
	std::cout << "Calcu sparsee" << std::endl;
	if (detectedLandmarks.size() < 1) {
		std::cout << "No face could be detected" << std::endl;
		return 1;
	}
	else if (detectedLandmarks.size() > 1) {
		std::cout << "Warning: More than one face has been detected" << std::endl;
	}
	Matrix4f Pose = GetPose(bfm, params, detectedLandmarks);
	std::cout << Pose << std::endl;
	//CalcSparseTerm(bfm.landmarks, params, detectedLandmarks);
#endif // DETECT_FACIAL_LANDMARKS
	return 0;
}
