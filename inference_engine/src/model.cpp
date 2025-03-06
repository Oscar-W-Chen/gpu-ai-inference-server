#include "model.h"
#include "cuda_utils.h"
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

namespace inference {

//====================================================
// Tensor Implementation
//====================================================

class Tensor::TensorImpl {
public:
    std::string name;
    DataType dtype;
    Shape shape;
    std::vector<uint8_t> cpu_data;
    void* gpu_data{nullptr};
    bool on_gpu{false};
    int device_id{0};

    ~TensorImpl() {
        // Free GPU memory if allocated
        if (gpu_data != nullptr) {
            cudaFree(gpu_data);
            gpu_data = nullptr;
        }
    }

    size_t ByteSize() const {
        size_t element_size = 0;
        switch (dtype) {
            case DataType::FLOAT32: element_size = 4; break;
            case DataType::INT32: element_size = 4; break;
            case DataType::INT64: element_size = 8; break;
            case DataType::UINT8: element_size = 1; break;
            case DataType::INT8: element_size = 1; break;
            case DataType::BOOL: element_size = 1; break;
            case DataType::FP16: element_size = 2; break;
            case DataType::STRING:
                throw std::runtime_error("String tensors don't have fixed byte size");
            default:
                throw std::runtime_error("Unknown data type");
        }
        return shape.NumElements() * element_size;
    }
};

Tensor::Tensor() : impl_(new TensorImpl()) {}

Tensor::Tensor(const std::string& name, DataType dtype, const Shape& shape)
    : impl_(new TensorImpl()) {
        impl_->name = name;
        impl_->dtype = dtype;
        impl_->shape = shape;

        // Allocate CPU memory for the tensor
        size_t byte_size = impl_->ByteSize();
        impl_->cpu_data.resize(byte_size);
}

Tensor::~Tensor() = default;

const std::string& Tensor::GetName() const {
    return impl_->name;
}

DataType Tensor::GetDataType() const {
    return impl_->dtype;
}

const Shape& Tensor::GetShape() const {
    return impl_->shape;
}

bool Tensor::Reshape(const Shape& new_shape){
    size_t old_elements = impl_->shape.NumElements();
    size_t new_elements = 1;
    for (auto dim : new_shape.dims) {
        new_elements *= dim;
    }
    if (old_elements != new_elements){
        // Need to resize the data buffer
        impl_->shape = new_shape;

        // Resize CPU data
        impl_->cpu_data.resize(impl_->ByteSize());

        // If on GPU, need to reallocate
        if (impl_->on_gpu && impl_->gpu_data != nullptr) {
            cudaFree(impl_->gpu_data);
            impl_->gpu_data = nullptr;

            //Allocate new GPU memory
            cudaError_t error = cudaMalloc(&impl_->gpu_data, impl_->ByteSize());
            if(error != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory: " 
                          << cudaGetErrorString(error) << std::endl;
                impl_->on_gpu = false;
                return false;
            }
        } else {
            // Can just update the shape
        impl_->shape = new_shape;
        }
    }
    return true;
}   

bool Tensor::toGPU(int device_id) {
    if (!cuda::IsCudaAvailable()) {
        std::cerr << "CUDA is not available" << std::endl;
        return false;
    }

    impl_->device_id = device_id;

    // Set device
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Already on GPU?
    if (impl_->on_gpu && impl_->gpu_data != nullptr){
        return true;
    }

    // Allocate GPU memory
    error = cudaMalloc(&impl_->gpu_data, impl_->ByteSize());
    if (error != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: "
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Copy data to GPU
    error = cudaMemcpy(
        impl_->gpu_data,
        impl_->cpu_data.data(),
        impl_->ByteSize(),
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data to GPU: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(impl_->gpu_data);
        impl_->gpu_data = nullptr;
        return false;
    }

    impl_->on_gpu = true;
    return true;
}

bool Tensor::toCPU() {
    if(!impl_->on_gpu || impl_->gpu_data == nullptr) {
        return true; // Already on CPU
    }

    // Set device
    cudaError_t error = cudaSetDevice(impl_->device_id);
    if (error != cudaSuccess) {
        std::cerr << "Failed to set device: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Copy data from GPU to CPU
    error = cudaMemcpy(
        impl_->cpu_data.data(),
        impl_->gpu_data,
        impl_->ByteSize(),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        std::cerr << "Failed to copy data from GPU: " 
                  << cudaGetErrorString(error) << std::endl;
        return false;
    }

    // Free GPU memory
    cudaFree(impl_->gpu_data);
    impl_->gpu_data = nullptr;
    impl_->on_gpu = false;

    return true;
}

// Template specializations for SetData and GetData would go here
// For brevity, we'll just implement float for now

template<>
bool Tensor::SetData(const std::vector<float>& data) {
    if(impl_->dtype != DataType::FLOAT32) {
        std::cerr << "Data type mismatch" << std::endl;
        return false;
    }

    if (data.size() != impl_->shape.NumElements()) {
        std::cerr << "Data size mismatch. Expected " 
                  << impl_->shape.NumElements() 
                  << " elements, got " 
                  << data.size() << std::endl;
        return false;
    }
    
    // Copy data to CPU buffer
    std::memcpy(impl_->cpu_data.data(), data.data(), impl_->ByteSize());

    // If on GPU, also update GPU memory
    if(impl_->on_gpu && impl_->gpu_data != nullptr) {
        cudaError_t error = cudaMemcpy(
            impl_->gpu_data,
            impl_->cpu_data.data(),
            impl_->ByteSize(),
            cudaMemcpyHostToDevice
        );

        if(error != cudaSuccess) {
            std::cerr << "Failed to update GPU data: "
                      << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }

    return true;
}

template<>
bool Tensor::GetData(std::vector<float>& data) const {
    if(impl_->dtype != DataType::FLOAT32) {
        std::cerr << "Data type mismatch" << std::endl;
        return false;
    }

    // If on GPU, sync to CPU first
    if (impl_->on_gpu && impl_->gpu_data != nullptr) {
        cudaError_t error = cudaMemcpy(
            const_cast<uint8_t>(impl_->cpu_data.data()),
            impl_->gpu_data,
            impl_->ByteSize(),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess) {
            std::cerr << "Failed to copy data from GPU: " 
                      << cudaGetErrorString(error) << std::endl;
            return false;
        }
    }

    // Resize output vector
    data.resize(impl_->shape.NumElements());

    // Copy data to output vector
    std::memcpy(data.data(), impl_->cpu_data.data(), impl_->ByteSize());

    return true;
}

//====================================================
// Model Implementation
//====================================================

class ModelImpl {
public:
    ModelImpl(const std::string& model_path,
              ModelType type,
              const ModelConfig& config,
              DeviceType device,
              int device_id)
        : model_path(model_path),
          type_(type),
          config_(config),
          device_(device),
          device_id_(device_id),
          loaded_(false) {
        
        // Initialize metadata
        metadata_.name = config_.name;
        metadata_.version = config_.version;
        metadata_.type = type_;
        metadata_.inputs = config_.input_names;
        metadata_.outputs = config_.output_names;
        metadata_.load_time_ns = 0;

        // Initialize stats
        stats_.inference_count = 0;
        stats_.total_inference_time_ns = 0;
        stats_.last_inference_time_ns = 0;
        stats_.memory_usage_bytes = 0;
    }

    ~ModelImpl(){
        if (loaded_) {
            Unload();
        }
    }

    bool Load() {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Check model file exists
        if (!FileExists(model_path_)) {
            last_error_ = "Model file not found: " + model_path_;
            return false;
        }

        // Model type specific loading would go here
        // For example, loading TensorFlow, TensorRT, ONNX, etc.
        // This is just a placeholder implementation

        switch (type_) {
            case ModelType::TENSORFLOW:
                loaded_ = LoadTensorFlowModel();
                break;
            
            case ModelType::TENSORRT:
                loaded_ = LoadTensorRTModel();
                break;
            
            case ModelType::ONNX:
                loaded_ = LoadONNXModel();
                break;
            
            case ModelType::PYTORCH:
                loaded_ = LoadPyTorchModel();
                break;

            case ModelType::CUSTOM:
                loaded_ = LoadCustomModel();
                break;
            
            default:
                last_error_ = "Unsupported model type";
                return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        metadata_.load_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();

        return loaded_;
    }

    bool Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        if (!loaded_) {
            last_error_ = "Model not loaded";
            return false;
        }
        
        // Validate inputs
        if (!ValidateInputs(inputs)) {
            return false;
        }
        
        // Prepare outputs
        if (!PrepareOutputs(outputs)) {
            return false;
        }

        // Model-specific inference would go here
        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = false;
        switch (type_) {
            case ModelType::TENSORFLOW:
                success = InferTensorFlow(inputs, outputs):
                break;
            
                case ModelType::TENSORRT:
                success = InferTensorRT(inputs, outputs);
                break;
                
            case ModelType::ONNX:
                success = InferONNX(inputs, outputs);
                break;
                
            case ModelType::PYTORCH:
                success = InferPyTorch(inputs, outputs);
                break;
                
            case ModelType::CUSTOM:
                success = InferCustom(inputs, outputs);
                break;
                
            default:
                last_error_ = "Unsupported model type";
                return false;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds(
            end_time - start_time).count();

        // Update stats
        stats_.inference_count++;
        stats_.total_inference_time_ns += duration;
        stats_.last_inference_time_ns = duration;

        return success;

        void Unload() {
            // Model-specific unloading would go here
            switch (type_) {
                case ModelType::TENSORFLOW:
                    UnloadTensorFlow();
                    break;
                    
                case ModelType::TENSORRT:
                    UnloadTensorRT();
                    break;
                    
                case ModelType::ONNX:
                    UnloadONNX();
                    break;
                    
                case ModelType::PYTORCH:
                    UnloadPyTorch();
                    break;
                    
                case ModelType::CUSTOM:
                    UnloadCustom();
                    break;
                    
                default:
                    break;
            }
            
            loaded_ = false;
        }

        ModelMetadata GetMetadata() const {
            return metadata_;
        }
        
        bool IsLoaded() const {
            return loaded_;
        }
        
        std::string GetLastError() const {
            return last_error_;
        }
        
        Model::Stats GetStats() const {
            return stats_;
        }
    }

private:
    // Model properties
    std::string model_path_;
    ModelType type_;
    ModelConfig config_;
    int device_id_;
    bool loaded_;
    std::string last_error_;

    // Model metadata
    ModelMetadata metadata_;

    // Performance statistics
    Model::Stats stats_;

    // Helper methods
    bool FileExists(const std::string& path) {
        std::ifstream file(path);
        return file.good();
    }

    bool ValidateInputs(const std::vector<Tensor>& inputs) {
        // Check input count
        if (inputs.size() != config_.input_names.size()) {
            last_error_ = "Expected " + std::to_string(config_.input_names.size()) +
                          " inputs, got " + std::to_string(inputs.size());
            return false;
        }

        for (const auto& input : inputs) {
            // Check name
            auto it = std::find(config_.input_names.begin(),
                               config_.input_names.end(),
                               input.GetName());
            if (it == config_.input_names.end()) {
                last_error_ = "Unexpected input name: " + input.GetName();
                return false;
            }

            // Check data type
            if (config_.input_types.count(input.GetName()) > 0 &&
                config_.input_types.at(input.GetName()) != input.GetDataType()) {
                last_error_ = "Input data type mismatch for " + input.GetName();
                return false;
            }

            // Check shape (if not dynamic)
            if (config_.input_shapes.count(input.GetName()) >) {
                const auto& expected_shape = config_.input_shapes.at(input.GetName());
                const auto& actual_shape = input.GetShape();

                // Check dimension count
                if (expected_shape.dims.size() != actual_shape.dims.size()) {
                    last_error_ = "Input shape mismatch for " + input.GetName() +
                                  ": expected " + std::to_string(expected_shape.dims.size()) +
                                  " dimensions, got " + std::to_string(actual_shape.dims.size());
                    return false;
                }

                // Check each dimension (allowing for dynamic dimensions with -1)
                for (size_t i = 0; i < expected_shape.dims.size(); i++) {
                    if (expected_shape.dims[i] != -1 &&
                        expected_shape.dims[i] != actual_shape.dims[i]) {
                        last_error_ = "Input shape mismatch for " + input.GetName() +
                                      " at dimension " + std::to_string(i);
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // Model-specific implementation placeholders
    // These would be implemented based on the actual model framework
    
    bool LoadTensorFlowModel() {
        // TODO: Implement TensorFlow model loading
        last_error_ = "TensorFlow model loading not implemented";
        return false;
    }
    
    bool LoadTensorRTModel() {
        // TODO: Implement TensorRT model loading
        last_error_ = "TensorRT model loading not implemented";
        return false;
    }
    
    bool LoadONNXModel() {
        // TODO: Implement ONNX model loading
        last_error_ = "ONNX model loading not implemented";
        return false;
    }
    
    bool LoadPyTorchModel() {
        // TODO: Implement PyTorch model loading
        last_error_ = "PyTorch model loading not implemented";
        return false;
    }
    
    bool LoadCustomModel() {
        // TODO: Implement custom model loading
        last_error_ = "Custom model loading not implemented";
        return false;
    }
    
    bool InferTensorFlow(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        // TODO: Implement TensorFlow inference
        last_error_ = "TensorFlow inference not implemented";
        return false;
    }
    
    bool InferTensorRT(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        // TODO: Implement TensorRT inference
        last_error_ = "TensorRT inference not implemented";
        return false;
    }
    
    bool InferONNX(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        // TODO: Implement ONNX inference
        last_error_ = "ONNX inference not implemented";
        return false;
    }
    
    bool InferPyTorch(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        // TODO: Implement PyTorch inference
        last_error_ = "PyTorch inference not implemented";
        return false;
    }
    
    bool InferCustom(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
        // TODO: Implement custom inference
        last_error_ = "Custom inference not implemented";
        return false;
    }
    
    void UnloadTensorFlow() {
        // TODO: Implement TensorFlow unloading
    }
    
    void UnloadTensorRT() {
        // TODO: Implement TensorRT unloading
    }
    
    void UnloadONNX() {
        // TODO: Implement ONNX unloading
    }
    
    void UnloadPyTorch() {
        // TODO: Implement PyTorch unloading
    }
    
    void UnloadCustom() {
        // TODO: Implement custom unloading
    }
};

//====================================================
// Model Public Methods
//====================================================

Model::Model(const std::string& model_path, 
    ModelType type,
    const ModelConfig& config,
    DeviceType device, 
    int device_id)
: impl_(new ModelImpl(model_path, type, config, device, device_id)) {}

Model::~Model() = default;

Model::Model(Model&& other) noexcept = default;
Model& Model::operator=(Model&& other) noexcept = default;

bool Model::Load() {
return impl_->Load();
}

bool Model::Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
return impl_->Infer(inputs, outputs);
}

ModelMetadata Model::GetMetadata() const {
return impl_->GetMetadata();
}

bool Model::IsLoaded() const {
return impl_->IsLoaded();
}

void Model::Unload() {
impl_->Unload();
}

std::string Model::GetLastError() const {
return impl_->GetLastError();
}

Model::Stats Model::GetStats() const {
return impl_->GetStats();
}

} // namespace inference