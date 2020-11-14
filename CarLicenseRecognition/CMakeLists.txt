cmake_minimum_required(VERSION 3.15)
project(CarLicenseRecognition)

set(CMAKE_CXX_STANDARD 14)
add_executable(CarLicenseRecognition main.cpp)

#修改为自己的路径
#set(OpenCV_DIR F:/xxx/OpenCV/mingw-build2)   windows 环境
# Mac 环境
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")


add_subdirectory(CarAnnTrain)
add_subdirectory(CarSvmTrain)
add_subdirectory(CarRecgnize)

target_link_libraries(CarLicenseRecognition ${OpenCV_LIBS})



