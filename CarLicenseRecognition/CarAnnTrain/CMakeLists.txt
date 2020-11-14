# CMakeList.txt: CarAnnTrain 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.15)
project(CarAnnTrain)

set(CMAKE_CXX_STANDARD 14)


add_executable(CarAnnTrain main.cpp utils.cpp  train.cpp)


#set(OpenCV_DIR F:/xxx/OpenCV/mingw-build2)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")


target_link_libraries( CarAnnTrain ${OpenCV_LIBS} )
