cmake_minimum_required(VERSION 3.16.3)

# set project name
project(Lane_detector_Using_OpenCV_for_autonomous_vehicle)

# config OpenCV
set(OpenCV_DIR "/usr/local/Cellar/opencv/4.5.4_3/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# set include directory (e.g. include header files)
set(final_proj_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include header files
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main src/demo.cpp src/LaneDetector.cpp)


# link dynamic libraries to this executable
target_link_libraries(main ${OpenCV_LIBS})
