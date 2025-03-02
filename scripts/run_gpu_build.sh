#!/bin/bash
# Script to build and test using the GPU Docker environment

set -e  # Exit immediately if a command fails

# Check for nvidia-docker/nvidia-container-runtime
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU driver not found or not accessible."
    echo "Please ensure NVIDIA drivers are installed and working."
    exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Please install Docker."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose not found. Please install Docker Compose."
    exit 1
fi

# Build and run the GPU build environment
echo "Building GPU build environment..."
docker-compose -f docker-compose.gpu.yml build

echo "Running build and tests in GPU environment..."
docker-compose -f docker-compose.gpu.yml up

# Check exit code
if [ $? -eq 0 ]; then
    echo "Build and tests completed successfully!"
    
    # Copy the built libraries from the Docker volume to the local filesystem
    echo "Copying built libraries to local filesystem..."
    CONTAINER_ID=$(docker-compose -f docker-compose.gpu.yml ps -q gpu-build)
    docker cp $CONTAINER_ID:/workspace/build ./
    
    echo "Build artifacts are available in the ./build directory"
else
    echo "Build or tests failed. See log output for details."
    exit 1
fi