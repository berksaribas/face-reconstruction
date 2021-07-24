#pragma once
#include <vector>
#include <dlib/image_processing/frontal_face_detector.h>
#include <Eigen/Eigen>
#include <FacialLandmarkDetection.h>
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
	cv::Mat image;
	double* depth;

	RGBD_Image(char * rgb_path,char * depth_path) {
		this->image = cv::imread(rgb_path);
		this->landmarks = DetectLandmarks(rgb_path, true, true);
		load_data(depth_path);
	};

	double get_depth(int x, int y) {
		return depth[x + y * 1920];

		if (x % 2 == 0 && y % 2 == 0) {
			return depth[x + y * 1920];
		}
		else if (x % 2 == 1 && y % 2 == 0) {
			double result = 0;
			int count = 0;
			if (depth[x - 1 + y * 1920] != 0) {
				result += depth[x - 1 + y * 1920];
				count++;
			}
			if (depth[x + 1 + y * 1920] != 0) {
				result += depth[x + 1 + y * 1920];
				count++;
			}
			if (count == 0) {
				return result;
			}
			return result / count;
		}
		else if (x % 2 == 0 && y % 2 == 1) {
			double result = 0;
			int count = 0;
			if (depth[x + (y - 1) * 1920] != 0) {
				result += depth[x + (y - 1) * 1920];
				count++;
			}
			if (depth[x + (y - 1) * 1920] != 0) {
				result += depth[x + (y - 1) * 1920];
				count++;
			}
			if (count == 0) {
				return result;
			}
			return result / count;
		}

		double result = 0;
		int count = 0;
		if (depth[x - 1 + (y - 1) * 1920] != 0) {
			result += depth[x - 1 + (y - 1) * 1920];
			count++;
		}
		if (depth[x - 1 + (y + 1) * 1920] != 0) {
			result += depth[x - 1 + (y + 1) * 1920];
			count++;
		}
		if (depth[x + 1 + (y - 1) * 1920] != 0) {
			result += depth[x + 1 + (y - 1) * 1920];
			count++;
		}
		if (depth[x + 1 + (y + 1) * 1920] != 0) {
			result += depth[x + 1 + (y + 1) * 1920];
			count++;
		}

		if (count == 0) {
			return result;
		}
		return result / count;		
	}
	//Method to load the binary data into a vector of 3D points where x and y are already projected into 2D	
	void load_data(const char * path){  
		std::ifstream inBinFile; 
		inBinFile.open(path, std::ios::out | std::ios::binary);
		// Define projection matrix from 3D to 2D
		//P matrix is in camera_info.yaml*/
		Eigen::Matrix<float, 3,4> P;
			P <<  1052.667867276341, 0, 962.4130834944134, 0, 0, 1052.020917785721, 536.2206151001486, 0, 0, 0, 1, 0;

		depth = new double[1920 * 1080];

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
					continue;
				}

				depth[(int)std::round(output[0]) + (int)std::round(output[1]) * 1920] = output[2];
			}
		}
		inBinFile.close();
	}
};