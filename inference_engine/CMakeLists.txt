cmake_minimum_required(VERSION 3.10)
project(inference_engine LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75)  # Target older architectures

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime installation")

# Check if ONNXRUNTIME_ROOT exists
if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNX Runtime directory not found at ${ONNXRUNTIME_ROOT}")
endif()

# Add ONNX Runtime include and library directories
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/cuda_utils.cu
    src/inference_bridge.cpp
    src/model.cpp
    src/inference_manager.cpp
    src/model_repository.cpp
)

# Create shared library
add_library(inference_engine SHARED ${SOURCES})

# Set output directory
set_target_properties(inference_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Link CUDA libraries
target_link_libraries(inference_engine 
    ${CUDA_LIBRARIES}
    onnxruntime)

# Install targets
install(TARGETS inference_engine
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# Install headers
install(DIRECTORY include/
    DESTINATION include/inference_engine
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "CUDA include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime path: ${ONNXRUNTIME_ROOT}")