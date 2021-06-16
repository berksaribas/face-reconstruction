#pragma once

#include <iostream>
#include <fstream>

#include "Eigen.h"
//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>
//#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

using namespace cv;
using namespace dlib;
using namespace std;

std::vector<full_object_detection> DetectLandmarks(char* imagePath="", bool presentLandmarks=false, bool holdImg=true) {
    std::vector<full_object_detection> empty;
    try
    {

        cv::VideoCapture cap(0); // 0
        if (imagePath != "")
            cap.open(imagePath); // 0

        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return empty; // 1
        }

        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

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
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;
            for (unsigned long i = 0; i < faces.size(); ++i)
                shapes.push_back(pose_model(cimg, faces[i]));

            for (unsigned long i = 0; i < faces.size(); ++i)
                for (unsigned long j = 0; j < pose_model(cimg, faces[i]).num_parts(); ++j)
                    cout << pose_model(cimg, faces[i]).part(j).x() << std::endl;

            if (presentLandmarks) {
                // Display it all on the screen
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(render_face_detections(shapes));

                if (holdImg) {
                    int key = waitKey(10000);
                    if (key == 27) {
                        break;
                    }
                }
            }

            return shapes;
        }
    }
    catch (serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
        return empty;
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
        return empty;
    }
}

/*try
{
    // This example takes in a shape model file and then a list of images to
    // process.  We will take these filenames in as command line arguments.
    // Dlib comes with example images in the examples/faces folder so give
    // those as arguments to this program.
    if (argc == 1)
    {
        cout << "Call this program like this:" << endl;
        cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
        cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
        cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        return 0;
    }

    // We need a face detector.  We will use this to get bounding boxes for
    // each face in an image.
    frontal_face_detector detector = get_frontal_face_detector();
    // And we also need a shape_predictor.  This is the tool that will predict face
    // landmark positions given an image and face bounding box.  Here we are just
    // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
    // as a command line argument.
    shape_predictor sp;
    deserialize(argv[1]) >> sp;


    image_window win, win_faces;
    // Loop over all the images provided on the command line.
    for (int i = 2; i < argc; ++i)
    {
        cout << "processing image " << argv[i] << endl;
        array2d<rgb_pixel> img;
        load_image(img, argv[i]);
        // Make the image larger so we can detect small faces.
        pyramid_up(img);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            cout << "number of parts: " << shape.num_parts() << endl;
            cout << "pixel position of first part:  " << shape.part(0) << endl;
            cout << "pixel position of second part: " << shape.part(1) << endl;
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            shapes.push_back(shape);
        }

        // Now let's view our face poses on the screen.
        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(render_face_detections(shapes));

        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        dlib::array<array2d<rgb_pixel> > face_chips;
        extract_image_chips(img, get_face_chip_details(shapes), face_chips);
        win_faces.set_image(tile_images(face_chips));

        cout << "Hit enter to process the next image..." << endl;
        cin.get();
    }
}
catch (exception& e)
{
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
}*/