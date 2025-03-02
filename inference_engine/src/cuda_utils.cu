#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>

namespace inference{
namespace cuda{

// CUDA kernel for vector addition
// This runs on the GPU - each thread handles one element
__global__ void addVectors(const float* a, const float* b, float* result, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

bool IsCudaAvailable(){
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess){
        std::cerr << "CUDA error checking for devices: "
                  << cudaGetErrorString(error) << std::endl;
        return false;  
    }

    return deviceCount > 0;
}

int GetDeviceCount(){
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess){
        std::cerr << "CUDA error getting device count: "
                  << cudaGetErrorString(error) << std::endl;
        return 0;
    }
    return deviceCount;
}

std::string GetDeviceInfo(int device_id) {
    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, device_id);
    if (error != cudaSuccess){
        std::cerr << "CUDA error getting device properties: "
                  << cudaGetErrorString(error) << std::endl;
        return "Unknown device";
    }

    // Format device information
    std::string info =  "Device " + std::to_string(device_id) + ": " + deviceProp.name +
                        " (Compute Capability " +
                        std::to_string(deviceProp.major) + "." +
                        std::to_string(deviceProp.minor) + ")";
    
    return info;
}

__host__ bool VectorAdd(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& result){
    // Validate input sizes
    if(a.size() != b.size()) {
        std::cerr << "Vector sizes does not match" << std::endl;
        return false;
    }
    int size = a.size();
    result.resize(size);

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;

    // Helper function to clean up resources
    auto cleanup = [&]() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_result) cudaFree(d_result);
    }

    // Allocate memory on GPU for first input vector
    cudaError_t error = cudaMalloc(&d_a, size * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr  << "Failed to allocate device memory for vector A: " 
                   << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }

    // Allocate memory on GPU for second input vector
    error = cudaMalloc(&d_b, size * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr  << "Failed to allocate device memory for vector B: " 
                   << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }

    // Allocate memory on GPU for result vector
    error = cudaMalloc(&d_result, size * sizeof(float));
    if (error != cudaSuccess) {
        std::cerr  << "Failed to allocate device memory for vector result: " 
                   << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }

    // Copy input data from host to device (CPU to GPU)
    error = cudaMemcpy(d_a, a.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std::cerr << "Failed to copy vector A to device: "
                  << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }
    
    error = cudaMemcpy(d_b, b.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess){
        std::cerr << "Failed to copy vector B to device: "
                  << cudaGetErrorString(error) << std::endl;
        cleanup();
        return false;
    }

    // Configure kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector addition kernel on GPU
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);

    // Wait for GPU to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr <<< "Kernel execution failed: "
                  <<< cudaGetErrorString(error) <<< std::endl;
        cleanup();
        return false;
    }

    // Copy the result back to host (GPU to CPU)
    error = cudaMemcpy(result.data(), d_result, size * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr <<< "Failed to copy result vector from device: "
                  <<< cudaGetErrorString(error) <<< std::endl;
        cleanup();
        return false;
    }

    // Free device memory
    cleanup();
    return true;
}

MemoryInfo GetMemoryInfo(int device_id){
    MemoryInfo memInfo = {0, 0, 0};

    // Set device
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device: " 
                  << cudaGetErrorString(error) << std::endl;
        return memInfo;
    }

    // Get memory info
    size_t free, total;
    error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess) {
        std::cerr << "Failed to get memory info: " 
                  << cudaGetErrorString(error) << std::endl;
        return memInfo;
    }

    memInfo.free = free;
    memInfo.total = total;
    memInfo.used = total - free;
    return memInfo;
}


}
}