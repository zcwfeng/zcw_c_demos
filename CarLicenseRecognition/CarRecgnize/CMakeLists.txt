cmake_minimum_required(VERSION 3.15)
project(CarRecgnize)

set(CMAKE_CXX_STANDARD 14)

add_executable(CarRecgnize main.cpp  CarPlateLocation.cpp CarColorPlateLocation.cpp CarSobelPlateLocation.cpp CarPlateRecgnize.cpp)

#set(OpenCV_DIR F:/xxx/OpenCV/mingw-build2)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "${OpenCV_INCLUDE_DIRS}")
message(STATUS "${OpenCV_LIBS}")


target_link_libraries( CarRecgnize ${OpenCV_LIBS} )
