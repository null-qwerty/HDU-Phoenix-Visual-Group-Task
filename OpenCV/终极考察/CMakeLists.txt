CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

set(CMAKE_CXX_STANDARD 11)
set(OpenCV_DIR /usr/local/lib/cmake/opencv4)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDES_DIRS})

project(final1)
add_executable(final1 1.cpp)
project(final2)
add_executable(final2 2.cpp)
project(final3)
add_executable(final3 3.cpp)

target_link_libraries(final1 ${OpenCV_LIBS})
target_link_libraries(final2 ${OpenCV_LIBS})
target_link_libraries(final3 ${OpenCV_LIBS})