cmake_minimum_required(VERSION 3.10)
project(inference_test LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Include inference engine directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../inference_engine/include)

# Test program
add_executable(cuda_test ${CMAKE_CURRENT_SOURCE_DIR}/../../test/cuda_test.cpp)

# Link with inference engine library
target_link_libraries(cuda_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine/lib/libinference_engine.so
    ${CUDA_LIBRARIES}
)
