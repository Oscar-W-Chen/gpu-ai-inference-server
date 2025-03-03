#include "cuda_utils.h"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "Testing CUDA Utilities" << std::endl;
    std::cout << "======================" << std::endl;

    // Check if CUDA is available
    bool cudaAvailable = inference::cuda::IsCudaAvailable();
    std::cout << "CUDA Available: " << (cudaAvailable ? "Yes" : "No") << std::endl;

    if (!cudaAvailable) {
        std::cout << "No CUDA-capable device detected. Exiting..." << std::endl;
        return 1;
    }

    // Get device count
    int deviceCount = inference::cuda::GetDeviceCount();
    std::cout << "CUDA Device Count: " << deviceCount << std::endl;

    for(int i = 0; i < deviceCount; i++){
        std::cout << inference::cuda::GetDeviceInfo(i) << std::endl;

        // Get memory info
        inference::cuda::MemoryInfo memInfo = inference::cuda::GetMemoryInfo(i);
        std::cout << " Memory Total: " << (memInfo.total / 1024 / 1024) << " MB" << std::endl;
        std::cout << " Memory Free: " << (memInfo.free / 1024 / 1024) << " MB" << std::endl;
        std::cout << " Memory Used: " << (memInfo.used / 1024 / 1024) << " MB" << std::endl;
    }

    // Test vector addition
    std::cout << "\nTesting Vector Addition" << std::endl;
    std::cout << "=========================" << std::endl;

    // Create test vectors
    const int vectorSize = 1000000;
    std::vector<float> a(vectorSize, 1.0f);
    std::vector<float> b(vectorSize, 1.0f);
    std::vector<float> result;

    // Perform vector addition
    bool success = inference::cuda::VectorAdd(a, b, result);

    if(success){
        // Verify results (check first 5 elements)
        std::cout << "Vector addition succeeded" << std::endl;
        
        std::cout << "Verifying first 5 elements:" << std::endl;
        for (int i = 0; i < 5 && result.size(); i++) {
            std::cout << a[i] << " + " << b[i] << " = " << result[i];
            if(result[i] == a[i] + b[i]){
                std::cout << " ✓" << std::endl;
            } else {
                std::cout << " ✗ (Expected: " << a[i] + b[i] << ")" << std::endl;
            }
        }
    } else {
        std::cout << "Vector addition failed" << std::endl;
    }

    return 0;
}