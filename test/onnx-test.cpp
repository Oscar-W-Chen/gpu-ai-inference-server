#include <iostream>
#include <vector>
#include <string>
#include "model.h"
#include "cuda_utils.h"

using namespace inference;

// Simple test to load and run inference on an ONNX model
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_directory>" << std::endl;
        return 1;
    }

    std::string model_directory = argv[1];
    std::cout << "Testing ONNX model loading and inference with model at: " << model_directory << std::endl;
    
    // Check CUDA availability
    bool cuda_available = cuda::IsCudaAvailable();
    std::cout << "CUDA available: " << (cuda_available ? "Yes" : "No") << std::endl;
    
    if (cuda_available) {
        int device_count = cuda::GetDeviceCount();
        std::cout << "CUDA device count: " << device_count << std::endl;
        
        for (int i = 0; i < device_count; i++) {
            std::cout << cuda::GetDeviceInfo(i) << std::endl;
            
            // Get memory info
            auto mem_info = cuda::GetMemoryInfo(i);
            std::cout << "Memory total: " << (mem_info.total / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "Memory free: " << (mem_info.free / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "Memory used: " << (mem_info.used / (1024 * 1024)) << " MB" << std::endl;
        }
    }
    
    // Create model configuration
    ModelConfig config;
    config.name = "test_model";
    config.version = "1";
    config.type = ModelType::ONNX;
    
    // Default input and output names
    // These should match the input/output tensor names in your ONNX model
    config.input_names = {"input"};
    config.output_names = {"output"};

    // Use GPU if available, otherwise CPU
    DeviceType device_type = cuda_available ? DeviceType::GPU : DeviceType::CPU;
    int device_id = 0;
    
    // Create the model
    Model model(model_directory, ModelType::ONNX, config, device_type, device_id);
    
    // Load the model
    std::cout << "\nLoading model..." << std::endl;
    bool load_success = model.Load();
    if (!load_success) {
        std::cerr << "Failed to load model: " << model.GetLastError() << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully!" << std::endl;
    
    // Get model metadata
    ModelMetadata metadata = model.GetMetadata();
    std::cout << "\nModel Metadata:" << std::endl;
    std::cout << "Name: " << metadata.name << std::endl;
    std::cout << "Version: " << metadata.version << std::endl;
    std::cout << "Type: " << static_cast<int>(metadata.type) << std::endl;
    std::cout << "Inputs: ";
    for (const auto& input : metadata.inputs) {
        std::cout << input << " ";
    }
    std::cout << std::endl;
    std::cout << "Outputs: ";
    for (const auto& output : metadata.outputs) {
        std::cout << output << " ";
    }
    std::cout << std::endl;
    
    // Create sample input tensor
    std::vector<Tensor> inputs;
    
    // For a simple test, create a 1x3 float tensor with all values set to 1.0
    // Adjust this based on your model's expected input shape
    Shape input_shape;
    input_shape.dims = {1, 3};
    Tensor input_tensor("input", DataType::FLOAT32, input_shape);
    
    // Fill with test data
    std::vector<float> input_data(input_shape.NumElements(), 1.0f);
    input_tensor.SetData(input_data);
    
    // Add to inputs
    inputs.push_back(input_tensor);
    
    // Prepare output vector
    std::vector<Tensor> outputs;
    
    // Run inference
    std::cout << "\nRunning inference..." << std::endl;
    bool infer_success = model.Infer(inputs, outputs);
    if (!infer_success) {
        std::cerr << "Failed to run inference: " << model.GetLastError() << std::endl;
        return 1;
    }
    
    // Print inference results
    std::cout << "Inference completed successfully!" << std::endl;
    std::cout << "\nOutput Tensors:" << std::endl;
    
    for (const auto& output : outputs) {
        std::cout << "Name: " << output.GetName() << std::endl;
        std::cout << "Data Type: " << static_cast<int>(output.GetDataType()) << std::endl;
        
        std::cout << "Shape: [";
        for (size_t i = 0; i < output.GetShape().dims.size(); i++) {
            std::cout << output.GetShape().dims[i];
            if (i < output.GetShape().dims.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        
        // For float tensors, print the first few values
        if (output.GetDataType() == DataType::FLOAT32) {
            std::vector<float> data;
            output.GetData(data);
            
            size_t print_count = std::min(size_t(10), data.size());
            std::cout << "First " << print_count << " values: ";
            for (size_t i = 0; i < print_count; i++) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    // Get model stats
    Model::Stats stats = model.GetStats();
    std::cout << "\nModel Statistics:" << std::endl;
    std::cout << "Inference Count: " << stats.inference_count << std::endl;
    std::cout << "Last Inference Time: " << (stats.last_inference_time_ns / 1000000.0) << " ms" << std::endl;
    std::cout << "Total Inference Time: " << (stats.total_inference_time_ns / 1000000.0) << " ms" << std::endl;
    std::cout << "Memory Usage: " << (stats.memory_usage_bytes / (1024 * 1024.0)) << " MB" << std::endl;
    
    // Unload the model
    std::cout << "\nUnloading model..." << std::endl;
    model.Unload();
    std::cout << "Model unloaded successfully!" << std::endl;
    
    return 0;
}