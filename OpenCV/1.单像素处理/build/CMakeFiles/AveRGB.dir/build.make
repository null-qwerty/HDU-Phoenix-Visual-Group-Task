# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build

# Include any dependencies generated for this target.
include CMakeFiles/AveRGB.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/AveRGB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/AveRGB.dir/flags.make

CMakeFiles/AveRGB.dir/main.cpp.o: CMakeFiles/AveRGB.dir/flags.make
CMakeFiles/AveRGB.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/AveRGB.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/AveRGB.dir/main.cpp.o -c /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/main.cpp

CMakeFiles/AveRGB.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/AveRGB.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/main.cpp > CMakeFiles/AveRGB.dir/main.cpp.i

CMakeFiles/AveRGB.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/AveRGB.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/main.cpp -o CMakeFiles/AveRGB.dir/main.cpp.s

# Object files for target AveRGB
AveRGB_OBJECTS = \
"CMakeFiles/AveRGB.dir/main.cpp.o"

# External object files for target AveRGB
AveRGB_EXTERNAL_OBJECTS =

AveRGB: CMakeFiles/AveRGB.dir/main.cpp.o
AveRGB: CMakeFiles/AveRGB.dir/build.make
AveRGB: /usr/local/lib/libopencv_calib3d.so.4.6.0
AveRGB: /usr/local/lib/libopencv_core.so.4.6.0
AveRGB: /usr/local/lib/libopencv_dnn.so.4.6.0
AveRGB: /usr/local/lib/libopencv_features2d.so.4.6.0
AveRGB: /usr/local/lib/libopencv_flann.so.4.6.0
AveRGB: /usr/local/lib/libopencv_highgui.so.4.6.0
AveRGB: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
AveRGB: /usr/local/lib/libopencv_imgproc.so.4.6.0
AveRGB: /usr/local/lib/libopencv_ml.so.4.6.0
AveRGB: /usr/local/lib/libopencv_objdetect.so.4.6.0
AveRGB: /usr/local/lib/libopencv_photo.so.4.6.0
AveRGB: /usr/local/lib/libopencv_stitching.so.4.6.0
AveRGB: /usr/local/lib/libopencv_video.so.4.6.0
AveRGB: /usr/local/lib/libopencv_videoio.so.4.6.0
AveRGB: /usr/local/lib/libopencv_imgcodecs.so.4.6.0
AveRGB: /usr/local/lib/libopencv_calib3d.so.4.6.0
AveRGB: /usr/local/lib/libopencv_dnn.so.4.6.0
AveRGB: /usr/local/lib/libopencv_features2d.so.4.6.0
AveRGB: /usr/local/lib/libopencv_flann.so.4.6.0
AveRGB: /usr/local/lib/libopencv_imgproc.so.4.6.0
AveRGB: /usr/local/lib/libopencv_core.so.4.6.0
AveRGB: CMakeFiles/AveRGB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable AveRGB"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/AveRGB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/AveRGB.dir/build: AveRGB

.PHONY : CMakeFiles/AveRGB.dir/build

CMakeFiles/AveRGB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/AveRGB.dir/cmake_clean.cmake
.PHONY : CMakeFiles/AveRGB.dir/clean

CMakeFiles/AveRGB.dir/depend:
	cd /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理 /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理 /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build /home/chenhy/桌面/RM/2023视觉组寒假任务/HDU-Phoenix-Visual-Group-Task/OpenCV/1.单像素处理/build/CMakeFiles/AveRGB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/AveRGB.dir/depend

