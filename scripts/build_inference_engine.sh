#!/bin/bash
# Script to build the inference engine library and tests with ONNX Runtime support
set -e # Exit immediately if a command fails

# Store the original directory to use for relative paths
ORIGINAL_DIR=$(pwd)

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
    
    # Hardcoded path to the test model, adjusted relative to the current directory
    # Use the ORIGINAL_DIR to compute the absolute path
    DEFAULT_MODEL_PATH="$ORIGINAL_DIR/models/test_model/1/model.onnx"
    
    # Allow override via command line
    MODEL_PATH="$DEFAULT_MODEL_PATH"
    if [[ "$*" == *--model-path=* ]]; then
        CUSTOM_PATH=$(echo "$*" | sed -n 's/.*--model-path=\([^ ]*\).*/\1/p')
        # If the custom path is absolute, use it directly; otherwise, compute relative to ORIGINAL_DIR
        if [[ "$CUSTOM_PATH" == /* ]]; then
            MODEL_PATH="$CUSTOM_PATH"
        else
            MODEL_PATH="$ORIGINAL_DIR/$CUSTOM_PATH"
        fi
    fi
    
    if [ -e "$MODEL_PATH" ]; then
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
        echo "Model file $MODEL_PATH not found. Skipping ONNX test."
        echo "Please run the following command to create a test model:"
        echo "python $ORIGINAL_DIR/scripts/create-test-model.py"
    fi
fi

# Return to the original directory
cd "$ORIGINAL_DIR"
echo "Build process completed successfully"