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
CMAKE_SOURCE_DIR = /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/FaceTrain.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FaceTrain.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FaceTrain.dir/flags.make

CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o: CMakeFiles/FaceTrain.dir/flags.make
CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o: ../CascadeDetectorAdapter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o -c /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/CascadeDetectorAdapter.cpp

CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/CascadeDetectorAdapter.cpp > CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.i

CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/CascadeDetectorAdapter.cpp -o CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.s

# Object files for target FaceTrain
FaceTrain_OBJECTS = \
"CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o"

# External object files for target FaceTrain
FaceTrain_EXTERNAL_OBJECTS =

FaceTrain: CMakeFiles/FaceTrain.dir/CascadeDetectorAdapter.cpp.o
FaceTrain: CMakeFiles/FaceTrain.dir/build.make
FaceTrain: /usr/local/lib/libopencv_gapi.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_stitching.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_alphamat.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_aruco.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_bgsegm.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_bioinspired.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_ccalib.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_dnn_objdetect.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_dnn_superres.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_dpm.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_face.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_freetype.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_fuzzy.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_hfs.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_img_hash.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_intensity_transform.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_line_descriptor.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_mcc.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_quality.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_rapid.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_reg.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_rgbd.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_saliency.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_sfm.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_stereo.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_structured_light.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_superres.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_surface_matching.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_tracking.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_videostab.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_viz.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_xfeatures2d.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_xobjdetect.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_xphoto.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_highgui.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_shape.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_datasets.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_plot.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_text.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_dnn.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_ml.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_phase_unwrapping.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_optflow.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_ximgproc.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_video.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_videoio.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_imgcodecs.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_objdetect.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_calib3d.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_features2d.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_flann.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_photo.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_imgproc.4.5.0.dylib
FaceTrain: /usr/local/lib/libopencv_core.4.5.0.dylib
FaceTrain: CMakeFiles/FaceTrain.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FaceTrain"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FaceTrain.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FaceTrain.dir/build: FaceTrain

.PHONY : CMakeFiles/FaceTrain.dir/build

CMakeFiles/FaceTrain.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FaceTrain.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FaceTrain.dir/clean

CMakeFiles/FaceTrain.dir/depend:
	cd /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug /Users/zcw/dev/c_workspace/zcw_c_demos/FaceTrain/cmake-build-debug/CMakeFiles/FaceTrain.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FaceTrain.dir/depend
