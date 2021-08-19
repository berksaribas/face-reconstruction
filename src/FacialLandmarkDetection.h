#pragma once

#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

static std::vector<dlib::full_object_detection> DetectLandmarks(char* imagePath="", bool presentLandmarks=false, bool holdImg=true) {
    std::vector<dlib::full_object_detection> empty;
    try
    {

        cv::VideoCapture cap(0); // 0
        if (imagePath != "")
            cap.open(imagePath); // 0

        if (!cap.isOpened())
        {
            std::cerr << "Unable to connect to camera\n";
            return empty; // 1
        }

        dlib::image_window win;

        // Load face detection and pose estimation models.
        dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        dlib::shape_predictor pose_model;
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        std::vector<dlib::full_object_detection> shapes;

        // Grab and process frames until the main window is closed by the user.
        while (!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            dlib::cv_image<dlib::bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<dlib::rectangle> faces = detector(cimg);
            // Find the pose of each face.
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));

            for (unsigned long i = 0; i < faces.size(); ++i)
                for (unsigned long j = 0; j < pose_model(cimg, faces[i]).num_parts(); ++j)
                    std::cout << pose_model(cimg, faces[i]).part(j).x() << std::endl;

            if (presentLandmarks) {
                // Display it all on the screen
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(dlib::render_face_detections(shapes));

                if (holdImg) {
                    int key = cv::waitKey(10000);
                    if (key == 27) {
                        break;
                    }
                }
            }
            else {
                break;
            }

            return shapes;
        }

        return shapes;
    }
    catch (dlib::serialization_error& e)
    {
        std::cout << "You need dlib's default face landmarking model file to run this example.\n";
        std::cout << "You can get it from the following URL: \n";
        std::cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n";
        std::cout << "\n" << e.what() << "\n";
        return empty;
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << "\n";
        return empty;
    }
}