cmake_minimum_required(VERSION 3.9)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(face_reconstruction)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")
set(glog_INCLUDE_DIR ${LIBRARY_DIR}/glog-lib/include CACHE PATH "Path to glog source folder")
set(Ceres_DIR ${LIBRARY_DIR}/Ceres/CMake CACHE PATH "Path to Ceres build folder")
set(Ceres_INCLUDE_DIR ${Ceres_DIR}/../include)
set(dlib_DIR ${LIBRARY_DIR}/Dlib/lib/cmake/dlib CACHE PATH "Path to Ceres build folder")
set(dlib_INCLUDE_DIR ${LIBRARY_DIR}/Dlib/include CACHE PATH "Path to glog source folder")
set(OpenCV_DIR "C:/opencv/build") #set opencv directory

# OpenGL stuff
set(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../glfw/include CACHE PATH "Path to glfw include")
set(GLFW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/../glfw/lib-vc2019/glfw3.lib CACHE PATH "Path to glfw include")

set(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../glew/include CACHE PATH "Path to glew include")
set(GLEW_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/../glew/lib/Release/x64/glew32s.lib CACHE PATH "Path to glew libraries")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${Ceres_DIR})

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)
find_package(glog REQUIRED ${glog_DIR_HINT})
find_package(dlib REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

add_definitions("-D_DISABLE_EXTENDED_ALIGNED_STORAGE")

if (MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /bigobj")
endif()

message(STATUS "ceres library name: ${Ceres_LIBRARIES}")
message(STATUS "path to ceres cmake information: ${Ceres_DIR}")
message(STATUS "path to ceres include directory: ${Ceres_INCLUDE_DIR}")
message(STATUS "glog library name: ${glog_LIBRARIES}")
message(STATUS "path to glog include directory = ${glog_INCLUDE_DIR}")

set(HEADER_FILES 
    BFM.h
    FacialLandmarkDetection.h
    Renderer.h
    DenseOptimizer.h
    Eigen.h
    RegularizationTerm.h
)
set(SOURCE_FILES 
    Renderer.cpp
    DenseOptimizer.cpp
)

add_executable(face_reconstruction main.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(face_reconstruction ceres dlib::dlib ${OpenCV_LIBS} ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARIES})
target_include_directories(face_reconstruction PUBLIC ${Eigen3_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}  ${Ceres_INCLUDE_DIR} ${glog_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${dlib_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS} ${GLFW_INCLUDE_DIRS})