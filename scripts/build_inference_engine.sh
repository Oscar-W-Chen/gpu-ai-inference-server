#!/bin/bash
# Script to build the inference engine library and tests
set -e # Exist immediately if a command fails

# Create necessary directories
echo "Create build directories..."
mkdir -p build/inference_engine
mkdir -p build/test

# Step 1: Build the inference engine library
echo "Building inference engine library..."
cd build/inference_engine
cmake ../../inference_engine -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "Inference engine library built successfully"

# Step 2: Build the CUDA test program
echo "Building CUDA test program..."
cd ../test
# Generate a temporary CMakeLists.txt for the test
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(inference_test LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(\${CUDA_INCLUDE_DIRS})

# Include inference engine directories
include_directories(\${CMAKE_CURRENT_SOURCE_DIR}/../../inference_engine/include)

# Test program
add_executable(cuda_test \${CMAKE_CURRENT_SOURCE_DIR}/../../test/cuda_test.cpp)

# Link with inference engine library
target_link_libraries(cuda_test
    \${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine/libinference_engine.so
    \${CUDA_LIBRARIES}
)
EOF

# Build the test
cmake .
make -j$(nproc)
echo "CUDA test program built successfully"

# Step 3: Run the CUDA test (only if --run-tests flag is provided)
if [[ "$*" == *--run-tests* ]]; then
    echo "Running CUDA tests..."
    ./cuda_test
    TEST_RESULT=$?
    if [ $TEST_RESULT -eq 0 ]; then
        echo "CUDA tests passed successfully!"
    else
        echo "CUDA tests failed with exit code $TEST_RESULT"
        exit $TEST_RESULT
    fi
fi

# Return to the original directory
cd ../..
echo "Build process completed successfully"