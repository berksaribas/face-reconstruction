#pragma once
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <Eigen/Eigen>
#include <stdio.h>
#include <BFM.h>
#include <iostream>     
#include <fstream>
#include <utils/points.h>
//#include <dlib/image_processing/render_face_detections.h>

/**
 * ICP optimizer - using Ceres for optimization.
 */
class RGBD_Image {
public:
	std::vector<dlib::full_object_detection> landmarks;
	std::vector<Point3D> points;
	// TODO: save the depth image

	RGBD_Image() {};

	double GetDepth(int x, int y) {
		// TODO: implement
		//landmarks[0].part(0).
		return 0.; 
	}
	//Method to load the binary data into a vector of 3D points where x and y are already projected into 2D	
	void load_data(const char * path){  
		std::ifstream inBinFile; 
		inBinFile.open(path, std::ios::out | std::ios::binary);
		// Define projection matrix from 3D to 2D
		//P matrix is in camera_info.yaml*/
		Eigen::Matrix<float, 3,4> P;
			P <<  1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

		for(int i =0; i<960;i++){
			for(int j =0; j<540;j++){
				double pointx;
				double pointy;
				double pointz;
				inBinFile.read(reinterpret_cast<char*>(&pointx), sizeof pointx);
				inBinFile.read(reinterpret_cast<char*>(&pointy), sizeof pointy);
				inBinFile.read(reinterpret_cast<char*>(&pointz), sizeof pointz);

				Eigen::Vector4f homogeneous_point(pointx, pointy, pointz,1);
				Eigen::Vector3f output = P * homogeneous_point;

				output[0] /= output[2];
				output[1] /= output[2];

				Point3D p{output[0],output[1],output[2]};
				//If there is no depth data at this pixel 
				if(std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)){   
					p.x = 0.0;
					p.y = 0.0;
					p.z = 0.0;
				}

				points.push_back(p);
			}
		}
		inBinFile.close();
	}
};