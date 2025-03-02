#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <string>
#include <vector>

namespace inference {
namespace cuda{

// Check if CUDA is available on the system
// Returns true if at least one CUDA-capable GPU is detected
bool IsCudaAvailable();

// Get the number of CUDA devices (GPUs) available
// Returns the count of available GPUs, or 0 if none
int GetDeviceCount();

// Get information about a specific CUDA device
// device_id: The ID of the device to query (default: 0 for the first GPU)
// Returns: A string with the device name
std::string GetDeviceInfo(int device_id=0);

// Simple CUDA vector addition to test basic functionality
// a: First input vector
// b: Second input vector
// result: Output vector that will contain a + b
// Returns true if successful, false otherwise
bool VectorAdd(const std::vector<float>& a,
               const std::vector<float>& b,
               std::vector<float>& result);

// Memory information structure to report GPU memory usage
struct MemoryInfo {
    size_t total;
    size_t free;
    size_t used;
};

// Get current memory usage for a specific device
MemoryInfo GetMemoryInfo(int device_id=0);


} // namespace cuda
} // namespace inference

#endif // CUDA_UTILS_H