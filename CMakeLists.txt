cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 87)
project(edgepoints)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
)

find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED PATHS /home/emnavi/.local/lib/python3.8/site-packages/torch/share/cmake/Torch)
message(STATUS "Found Torch include is ${TORCH_INCLUDE_DIRS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# TensorRT paths 
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

# Include directories
include_directories(include/ ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(edgepoints_lib SHARED
    src/edgepoints.cpp
    include/edgepoints.h 
    include/InterpolateSparse2D.h
    include/utils.h
)
link_directories(${CUDA_LIBRARY_DIRS} ${TENSORRT_LIBRARY_DIR})
target_link_libraries(edgepoints_lib ${catkin_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TORCH_LIBRARIES} nvinfer nvonnxparser nvinfer_plugin yaml-cpp)

add_executable(demo src/demo.cpp)
target_link_libraries(demo ${catkin_LIBRARIES} edgepoints_lib)