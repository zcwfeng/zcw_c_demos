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
include CarRecgnize/CMakeFiles/CarRecgnize.dir/depend.make

# Include the progress variables for this target.
include CarRecgnize/CMakeFiles/CarRecgnize.dir/progress.make

# Include the compile flags for this target's objects.
include CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make

CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.o: CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make
CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.o: ../CarRecgnize/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarRecgnize.dir/main.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/main.cpp

CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarRecgnize.dir/main.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/main.cpp > CMakeFiles/CarRecgnize.dir/main.cpp.i

CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarRecgnize.dir/main.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/main.cpp -o CMakeFiles/CarRecgnize.dir/main.cpp.s

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o: CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make
CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o: ../CarRecgnize/CarPlateLocation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateLocation.cpp

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateLocation.cpp > CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.i

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateLocation.cpp -o CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.s

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o: CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make
CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o: ../CarRecgnize/CarColorPlateLocation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarColorPlateLocation.cpp

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarColorPlateLocation.cpp > CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.i

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarColorPlateLocation.cpp -o CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.s

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o: CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make
CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o: ../CarRecgnize/CarSobelPlateLocation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarSobelPlateLocation.cpp

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarSobelPlateLocation.cpp > CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.i

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarSobelPlateLocation.cpp -o CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.s

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o: CarRecgnize/CMakeFiles/CarRecgnize.dir/flags.make
CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o: ../CarRecgnize/CarPlateRecgnize.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateRecgnize.cpp

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.i"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateRecgnize.cpp > CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.i

CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.s"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize/CarPlateRecgnize.cpp -o CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.s

# Object files for target CarRecgnize
CarRecgnize_OBJECTS = \
"CMakeFiles/CarRecgnize.dir/main.cpp.o" \
"CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o" \
"CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o" \
"CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o" \
"CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o"

# External object files for target CarRecgnize
CarRecgnize_EXTERNAL_OBJECTS =

CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/main.cpp.o
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateLocation.cpp.o
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/CarColorPlateLocation.cpp.o
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/CarSobelPlateLocation.cpp.o
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/CarPlateRecgnize.cpp.o
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/build.make
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_gapi.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_stitching.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_alphamat.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_aruco.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_bgsegm.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_bioinspired.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_ccalib.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_dnn_objdetect.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_dnn_superres.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_dpm.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_face.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_freetype.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_fuzzy.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_hfs.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_img_hash.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_intensity_transform.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_line_descriptor.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_mcc.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_quality.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_rapid.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_reg.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_rgbd.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_saliency.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_sfm.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_stereo.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_structured_light.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_superres.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_surface_matching.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_tracking.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_videostab.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_viz.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_xfeatures2d.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_xobjdetect.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_xphoto.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_highgui.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_shape.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_datasets.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_plot.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_text.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_dnn.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_ml.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_phase_unwrapping.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_optflow.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_ximgproc.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_video.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_videoio.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_imgcodecs.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_objdetect.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_calib3d.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_features2d.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_flann.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_photo.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_imgproc.4.5.0.dylib
CarRecgnize/CarRecgnize: /usr/local/lib/libopencv_core.4.5.0.dylib
CarRecgnize/CarRecgnize: CarRecgnize/CMakeFiles/CarRecgnize.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable CarRecgnize"
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CarRecgnize.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CarRecgnize/CMakeFiles/CarRecgnize.dir/build: CarRecgnize/CarRecgnize

.PHONY : CarRecgnize/CMakeFiles/CarRecgnize.dir/build

CarRecgnize/CMakeFiles/CarRecgnize.dir/clean:
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize && $(CMAKE_COMMAND) -P CMakeFiles/CarRecgnize.dir/cmake_clean.cmake
.PHONY : CarRecgnize/CMakeFiles/CarRecgnize.dir/clean

CarRecgnize/CMakeFiles/CarRecgnize.dir/depend:
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/CarRecgnize /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize /Users/zcw/dev/c_workspace/zcw_c_demos/CarLicenseRecognition/cmake-build-debug/CarRecgnize/CMakeFiles/CarRecgnize.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CarRecgnize/CMakeFiles/CarRecgnize.dir/depend

