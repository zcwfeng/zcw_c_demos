# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug

# Include any dependencies generated for this target.
include CarAnnTrain/CMakeFiles/CarAnnTrain.dir/depend.make

# Include the progress variables for this target.
include CarAnnTrain/CMakeFiles/CarAnnTrain.dir/progress.make

# Include the compile flags for this target's objects.
include CarAnnTrain/CMakeFiles/CarAnnTrain.dir/flags.make

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.o: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/flags.make
CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.o: ../CarAnnTrain/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarAnnTrain.dir/main.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/main.cpp

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarAnnTrain.dir/main.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/main.cpp > CMakeFiles/CarAnnTrain.dir/main.cpp.i

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarAnnTrain.dir/main.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/main.cpp -o CMakeFiles/CarAnnTrain.dir/main.cpp.s

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.o: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/flags.make
CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.o: ../CarAnnTrain/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarAnnTrain.dir/utils.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/utils.cpp

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarAnnTrain.dir/utils.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/utils.cpp > CMakeFiles/CarAnnTrain.dir/utils.cpp.i

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarAnnTrain.dir/utils.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/utils.cpp -o CMakeFiles/CarAnnTrain.dir/utils.cpp.s

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.o: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/flags.make
CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.o: ../CarAnnTrain/train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarAnnTrain.dir/train.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/train.cpp

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarAnnTrain.dir/train.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/train.cpp > CMakeFiles/CarAnnTrain.dir/train.cpp.i

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarAnnTrain.dir/train.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain/train.cpp -o CMakeFiles/CarAnnTrain.dir/train.cpp.s

# Object files for target CarAnnTrain
CarAnnTrain_OBJECTS = \
"CMakeFiles/CarAnnTrain.dir/main.cpp.o" \
"CMakeFiles/CarAnnTrain.dir/utils.cpp.o" \
"CMakeFiles/CarAnnTrain.dir/train.cpp.o"

# External object files for target CarAnnTrain
CarAnnTrain_EXTERNAL_OBJECTS =

CarAnnTrain/CarAnnTrain: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/main.cpp.o
CarAnnTrain/CarAnnTrain: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/utils.cpp.o
CarAnnTrain/CarAnnTrain: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/train.cpp.o
CarAnnTrain/CarAnnTrain: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/build.make
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_gapi.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_stitching.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_alphamat.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_aruco.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_bgsegm.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_bioinspired.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_ccalib.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_dnn_objdetect.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_dnn_superres.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_dpm.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_face.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_freetype.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_fuzzy.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_hfs.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_img_hash.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_intensity_transform.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_line_descriptor.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_mcc.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_quality.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_rapid.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_reg.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_rgbd.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_saliency.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_sfm.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_stereo.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_structured_light.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_superres.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_surface_matching.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_tracking.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_videostab.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_viz.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_xfeatures2d.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_xobjdetect.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_xphoto.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_highgui.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_shape.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_datasets.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_plot.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_text.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_dnn.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_ml.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_phase_unwrapping.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_optflow.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_ximgproc.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_video.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_videoio.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_imgcodecs.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_objdetect.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_calib3d.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_features2d.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_flann.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_photo.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_imgproc.4.5.0.dylib
CarAnnTrain/CarAnnTrain: /usr/local/lib/libopencv_core.4.5.0.dylib
CarAnnTrain/CarAnnTrain: CarAnnTrain/CMakeFiles/CarAnnTrain.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable CarAnnTrain"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CarAnnTrain.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CarAnnTrain/CMakeFiles/CarAnnTrain.dir/build: CarAnnTrain/CarAnnTrain

.PHONY : CarAnnTrain/CMakeFiles/CarAnnTrain.dir/build

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/clean:
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain && $(CMAKE_COMMAND) -P CMakeFiles/CarAnnTrain.dir/cmake_clean.cmake
.PHONY : CarAnnTrain/CMakeFiles/CarAnnTrain.dir/clean

CarAnnTrain/CMakeFiles/CarAnnTrain.dir/depend:
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarAnnTrain /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarAnnTrain/CMakeFiles/CarAnnTrain.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CarAnnTrain/CMakeFiles/CarAnnTrain.dir/depend

