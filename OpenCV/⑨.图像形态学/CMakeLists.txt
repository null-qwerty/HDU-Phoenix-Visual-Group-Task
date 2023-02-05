CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR /usr/local/bin/cmake/opencv4)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDES_DIRS})

project(morphology)
add_executable(morphology main.cpp)

target_link_libraries(morphology ${OpenCV_LIBS})