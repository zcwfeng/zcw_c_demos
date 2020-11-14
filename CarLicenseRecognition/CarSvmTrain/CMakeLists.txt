cmake_minimum_required(VERSION 3.15)
project(CarSvmTrain)

set(CMAKE_CXX_STANDARD 14)

add_executable(CarSvmTrain main.cpp train.cpp utils.cpp)

#set(OpenCV_DIR F:/xxx/OpenCV/mingw-build2)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")


target_link_libraries( CarSvmTrain ${OpenCV_LIBS} )