#!/bin/bash
# Script to build the inference engine library and tests with ONNX Runtime support
set -e # Exit immediately if a command fails

# Check if ONNX Runtime is installed
if [ ! -d "/usr/local/onnxruntime" ]; then
    echo "ONNX Runtime not found. Installing..."
    ./scripts/setup-onnxruntime.sh
else
    echo "ONNX Runtime installation detected."
fi

# Create necessary directories
echo "Creating build directories..."
mkdir -p build/inference_engine
mkdir -p build/test

# Step 1: Build the inference engine library
echo "Building inference engine library..."
cd build/inference_engine
cmake ../../inference_engine -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/usr/local/onnxruntime
make -j$(nproc)
echo "Inference engine library built successfully"

# Step 2: Build the ONNX test program
echo "Building ONNX test program..."
cd ../test
# Generate a temporary CMakeLists.txt for the test
cat > CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(inference_test LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(\${CUDA_INCLUDE_DIRS})

# ONNX Runtime configuration
set(ONNXRUNTIME_ROOT "/usr/local/onnxruntime" CACHE PATH "Path to ONNX Runtime installation")
include_directories(\${ONNXRUNTIME_ROOT}/include)
link_directories(\${ONNXRUNTIME_ROOT}/lib)

# Include inference engine directories
include_directories(\${CMAKE_CURRENT_SOURCE_DIR}/../../inference_engine/include)

# Test programs
add_executable(onnx_test \${CMAKE_CURRENT_SOURCE_DIR}/../../test/onnx_test.cpp)
add_executable(cuda_test \${CMAKE_CURRENT_SOURCE_DIR}/../../test/cuda_test.cpp)

# Link with inference engine library
target_link_libraries(onnx_test
    \${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine/lib/libinference_engine.so
    \${CUDA_LIBRARIES}
    onnxruntime
)

target_link_libraries(cuda_test
    \${CMAKE_CURRENT_SOURCE_DIR}/../inference_engine/lib/libinference_engine.so
    \${CUDA_LIBRARIES}
)
EOF

# Build the test
cmake .
make -j$(nproc)
echo "ONNX test program built successfully"

# Step 3: Run the CUDA test (only if --run-tests flag is provided)
if [[ "$*" == *--run-tests* ]]; then
    echo "Running CUDA tests..."
    ./cuda_test
    CUDA_TEST_RESULT=$?
    if [ $CUDA_TEST_RESULT -eq 0 ]; then
        echo "CUDA tests passed successfully!"
    else
        echo "CUDA tests failed with exit code $CUDA_TEST_RESULT"
        exit $CUDA_TEST_RESULT
    fi
    
    # If a model path is provided, run the ONNX test too
    if [[ "$*" == *--model-path=* ]]; then
        MODEL_PATH=$(echo "$*" | sed -n 's/.*--model-path=\([^ ]*\).*/\1/p')
        if [ -d "$MODEL_PATH" ]; then
            echo "Running ONNX tests with model at $MODEL_PATH..."
            ./onnx_test "$MODEL_PATH"
            ONNX_TEST_RESULT=$?
            if [ $ONNX_TEST_RESULT -eq 0 ]; then
                echo "ONNX tests passed successfully!"
            else
                echo "ONNX tests failed with exit code $ONNX_TEST_RESULT"
                exit $ONNX_TEST_RESULT
            fi
        else
            echo "Model path $MODEL_PATH does not exist. Skipping ONNX test."
        fi
    else
        echo "No model path provided. To run ONNX test, use --model-path=<path>"
    fi
fi

# Return to the original directory
cd ../..
echo "Build process completed successfully"