#include "model.h"
#include "cuda_utils.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring> // For memcpy
#include <cuda_runtime.h> // Need to include cuda_runtime.h for CUDA functions

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

// Copy constructor
Tensor::Tensor(const Tensor& other) : impl_(new TensorImpl(*other.impl_)) {}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept : impl_(std::move(other.impl_)) {}

// Copy assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        impl_ = std::make_unique<TensorImpl>(*other.impl_);
    }
    return *this;
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
            const_cast<uint8_t*>(impl_->cpu_data.data()),
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
        : model_path_(model_path),
          type_(type),
          config_(config),
          device_type_(device),
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

    ~ModelImpl() {
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
        
        // TODO: Prepare outputs
        // Placeholder for now to avoid the function call
        // if (!PrepareOutputs(outputs)) {
        //    return false;
        // }

        // Model-specific inference would go here
        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = false;
        switch (type_) {
            case ModelType::TENSORFLOW:
                success = InferTensorFlow(inputs, outputs);
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
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time).count();

        // Update stats
        stats_.inference_count++;
        stats_.total_inference_time_ns += duration;
        stats_.last_inference_time_ns = duration;

        return success;
    }

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

private:
    // Model properties
    std::string model_path_;
    ModelType type_;
    ModelConfig config_;
    DeviceType device_type_;
    int device_id_;
    bool loaded_;
    std::string last_error_;

    // Model metadata
    ModelMetadata metadata_;

    // Performance statistics
    Model::Stats stats_;

    //ONNX Runtime specific members
    ORT::Env onnx_env;
    std::unique_ptr<Ort::Session> onnx_session_;
    std::vector<std::string> onnx_input_names_;
    std::vector<std::string> onnx_output_names_;
    std::vector<std::vector<int64_t>> onnx_input_shapes_;
    std::vector<std::vector<int64_t>> onnx_output_shapes_;
    std::vector<ONNXTensorElementDataType> onnx_input_types_;
    std::vector<ONNXTensorElementDataType> onnx_output_types_;

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
            if (config_.input_shapes.count(input.GetName()) > 0) {
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
        try {
            std::cout << "Loading ONNX model: " << model_path_ << std::endl;

            // Create ONNX Runtime environment
            ORT::Env onnx_env_ = ORT::Env(ORT_LOGGING_LEVEL_WARNING, "inference-server");

            // Create session options
            Ort::SessionOptions session_options;

            // Enable CUDA if available and requested
            if (device_type_ == DeviceType::GPU && cuda::IsCudaAvailable()) {
                // Register CUDA provider
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = device_id_;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.gpu_mem_limit = 0; // No limit
                cuda_options.arena_extend_strategy = 0; // Default strategy
                cuda_options.do_copy_in_default_stream = 1;

                session_options.AppendExecutionProvider_CUDA(cuda_options);

                std::cout << "CUDA execution provider registered for ONNX Runtime" << std::endl;
            }

            // Set graph optimization level
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Set intra-op threat count
            session_options.SetIntraOpNumThreads(1);

            // Set execution mode
            session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

            std::ifstream file(model_file_path, std::ios::binary);
            if (!file.good()) {
                last_error_ = "ONNX model file not found: " + model_file_path;
                return false;
            }
            file.close();

            // Create session
            std::cout << "Creating ONNX Runtime session for " << model_file_path << std::end;
            onnx_session_ = std::make_unique<Ort::Session>(onnx_env_, model_file_path.c_str(), session_options);

            // Get model metadata
            Ort::AllocatorWithDefaultOptions allocator;

            // Get number of inputs and outputs
            size_t num_inputs = onnx_session_->GetInputCount();
            size_t num_outputs = onnx_session_->GetOutputCount();

            std::cout << "Model has " << num_inputs << " inputs and " << num_outputs << " outputs" << std::endl;

            // Clear previous data
            onnx_input_names_.clear();
            onnx_output_names_.clear();
            onnx_input_shapes_.clear();
            onnx_output_shapes_.clear();
            onnx_input_types_.clear();
            onnx_output_types_.clear();

            // Get input information
            onnx_input_names_.resize(num_inputs);
            onnx_input_shapes_.resize(num_inputs);
            onnx_input_types_.resize(num_inputs);

            for (size_t i = 0; i < num_inputs; i++) {
                // Get input name
                auto input_name = onnx_session_->GetInputNameAllocated(i, allocator);
                onnx_input_names_[i] = input_name.get();
                
                // Get input type
                Ort::TypeInfo type_info = onnx_session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                onnx_input_types_[i] = tensor_info.GetElementType();
                
                // Get input shape
                onnx_input_shapes_[i] = tensor_info.GetShape();
                
                // Print input info
                std::cout << "Input " << i << ": " << onnx_input_names_[i] << ", Type: " << onnx_input_types_[i] << std::endl;
                std::cout << "  Shape: [";
                for (size_t j = 0; j < onnx_input_shapes_[i].size(); j++) {
                    std::cout << onnx_input_shapes_[i][j];
                    if (j < onnx_input_shapes_[i].size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
            }

            // Get output information
            onnx_output_names_.resize(num_outputs);
            onnx_output_shapes_.resize(num_outputs);
            onnx_output_types_.resize(num_outputs);

            for (size_t i = 0; i < num_outputs; i++) {
                // Get output name
                auto output_name = onnx_session_->GetOutputNameAllocated(i, allocator);
                onnx_output_names_[i] = output_name.get();
                
                // Get output type
                Ort::TypeInfo type_info = onnx_session_->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                onnx_output_types_[i] = tensor_info.GetElementType();
                
                // Get output shape
                onnx_output_shapes_[i] = tensor_info.GetShape();
                
                // Print output info
                std::cout << "Output " << i << ": " << onnx_output_names_[i] << ", Type: " << onnx_output_types_[i] << std::endl;
                std::cout << "  Shape: [";
                for (size_t j = 0; j < onnx_output_shapes_[i].size(); j++) {
                    std::cout << onnx_output_shapes_[i][j];
                    if (j < onnx_output_shapes_[i].size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
            }

            // Update memory usage statistics
            stats_.memory_usage_bytes = EstimateModeMemoryUsage();

            std::cout << "ONNX model loaded successfully" << std::endl;
            return true;
        } catch (const Ort::Exception& e) {
            last_error_ = std::string("ONNX Runtime error: ") + e.what();
            std::cerr << last_error_ << std::endl;
            return false;
        } catch (const std::exception& e) {
            last_error_ = std::string("ONNX model loading error: ") + e.what();
            std::cerr << last_error_ << std::endl;
            return false;
        }
    }

    // Helper function to convert from ONNX data type to inference::DataType
    DataType ConvertFromOnnxDataType(ONNXTensorElementDataType onnx_type) {
        switch (onnx_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DataType::FLOAT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DataType::INT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DataType::INT64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DataType::UINT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return DataType::INT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: return DataType::STRING;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DataType::BOOL;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::FP16;
            default: return DataType::UNKNOWN;
        }
    }

    // Helper function to convert from inference::DataType to ONNX data type
    ONNXTensorElementDataType ConvertToOnnxDataType(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case DataType::INT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            case DataType::INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            case DataType::UINT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case DataType::INT8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
            case DataType::STRING: return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
            case DataType::BOOL: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            case DataType::FP16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        }
    }

    // Helper function to estimate model memory usage
    size_t EstimateModeMemoryUsage() const {
        // Simplified estimation to calculate ONNX Runtime memory usage
        size_t total_size = 0;

        // Sum up input tensor sizes
        for (size_t i = 0; i < onnx_input_shapes_.size(); i++) {
            size_t tensor_size = 1;
            for (const auto& dim : onnx_input_shapes_[i]) {
                if (dim > 0) { // skip dynamic dimensions
                    tensor_size *= dim;
                }
            }

            // Multiply by element size based on data type
            switch (onnx_input_types_[i]) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    tensor_size *=4;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    tensor_size *= 8;
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    tensor_size *= 1;  // 1 byte
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    tensor_size *= 2;  // 2 bytes
                    break;
                default:
                    tensor_size *= 4;  // Default to 4 bytes
            }

            total_size += tensor_size;
        }

        // Sum up output tensor sizes
        for (size_t i = 0; i < onnx_output_shapes_.size(); i++) {
            size_t tensor_size = 1;
            for (const auto& dim : onnx_output_shapes_[i]) {
                if (dim > 0) {  // Skip dynamic dimensions
                    tensor_size *= dim;
                }
            }
            
            // Multiply by element size based on data type
            switch (onnx_output_types_[i]) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    tensor_size *= 4;  // 4 bytes
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    tensor_size *= 8;  // 8 bytes
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    tensor_size *= 1;  // 1 byte
                    break;
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    tensor_size *= 2;  // 2 bytes
                    break;
                default:
                    tensor_size *= 4;  // Default to 4 bytes
            }
            
            total_size += tensor_size;
        }

        // Add a base memory overhead for the model weights and runtime
        // This is a rough estimation - actual overhead depends on the model
        constexpr size_t BASE_OVERHEAD_BYTES = 10 * 1024 * 1024; // 10 MB
        return total_size + BASE_OVERHEAD_BYTES;
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
        if (!onnx_session_) {
            last_error_ = "ONNX model is not loaded";
            return false;
        }

        try {
            std::cout << "Running inference with ONNX model" << std::endl;

            auto start_time = std::chrono::high_resolution_clock::now();

            // Create memory info
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            // Prepare input tensors
            std::vector<Ort::Value> ort_inputs;

            // Map input tensor names to input indices
            std::unordered_map<std::string, size_t> input_name_to_index;
            for (size_t i = 0; i < onnx_input_names_.size(); i++) {
                input_name_to_index[onnx_input_names_[i]] = i;
            }
            
            // Check that we have all required inputs
            for (const auto& input_name : onnx_input_names_) {
                bool found = false;
                for (const auto& input : inputs) {
                    if (input.GetName() == input_name) {
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    last_error_ = "Required input tensor not provided: " + input_name;
                    return false;
                }
            }

            // Convert inputs to ONNX format
            ort_inputs.reserve(inputs.size());
            for (const auto& input : inputs) {
                // Find the index for this input name
                auto it = input_name_to_index.find(input.GetName());
                if (it == input_name_to_index.end()) {
                    // Skip inputs that are not required by the model
                    continue;
                }
                
                size_t input_idx = it->second;
                
                // Get input shape and convert to int64_t
                const auto& shape = input.GetShape();
                std::vector<int64_t> input_shape(shape.dims.begin(), shape.dims.end());
                
                // Get the element count
                size_t element_count = shape.NumElements();
                
                // Create ONNX tensor based on data type
                DataType dtype = input.GetDataType();
    
                switch (dtype) {
                    case DataType::FLOAT32: {
                        std::vector<float> data;
                        if (!input.GetData(data)) {
                            last_error_ = "Failed to get FLOAT32 data for input: " + input.GetName();
                            return false;
                        }
                        
                        if (data.size() != element_count) {
                            last_error_ = "Input data size mismatch for " + input.GetName();
                            return false;
                        }
                        
                        Ort::Value tensor = Ort::Value::CreateTensor<float>(
                            memory_info, data.data(), data.size(),
                            input_shape.data(), input_shape.size()
                        );
                        ort_inputs.push_back(std::move(tensor));
                        break;
                    }
    
                    case DataType::INT32: {
                        std::vector<int32_t> data;
                        if (!input.GetData(data)) {
                            last_error_ = "Failed to get INT32 data for input: " + input.GetName();
                            return false;
                        }
                        
                        if (data.size() != element_count) {
                            last_error_ = "Input data size mismatch for " + input.GetName();
                            return false;
                        }
                        
                        Ort::Value tensor = Ort::Value::CreateTensor<int32_t>(
                            memory_info, data.data(), data.size(),
                            input_shape.data(), input_shape.size()
                        );
                        ort_inputs.push_back(std::move(tensor));
                        break;
                    }
    
                    // Add cases for other data types as needed
                    
                    default:
                        last_error_ = "Unsupported data type for input: " + input.GetName();
                        return false;
                }
            }

            // Order input tensors according to model's expected order
            std::vector<Ort::Value> ordered_inputs;
            ordered_inputs.resize(onnx_input_names_.size());
            
            for (size_t i = 0; i < inputs.size(); i++) {
                auto it = input_name_to_index.find(inputs[i].GetName());
                if (it != input_name_to_index.end()) {
                    size_t model_input_idx = it->second;
                    if (i < ort_inputs.size()) {
                        ordered_inputs[model_input_idx] = std::move(ort_inputs[i]);
                    }
                }
            }

            // Prepare output names
            std::vector<const char*> output_names_cstr;
            output_names_cstr.reserve(onnx_output_names_.size());
            for (const auto& name : onnx_output_names_) {
                output_names_cstr.push_back(name.c_str());
            }

            // Convert input names to C strings
            std::vector<const char*> input_names_cstr;
            input_names_cstr.reserve(onnx_input_names_.size());
            for (const auto& name : onnx_input_names_) {
                input_names_cstr.push_back(name.c_str());
            }

            // Run inference
            auto ort_outputs = onnx_session_->Run(
                Ort::RunOptions{nullptr},
                input_names_cstr.data(),
                ordered_inputs.data(),
                ordered_inputs.size(),
                output_names_cstr.data(),
                output_names_cstr.size()
            );

            // Prepare outputs
            outputs.clear();
            outputs.reserve(ort_outputs.size());

            // Convert ONNX outputs to our tensor format
            for (size_t i = 0; i < ort_outputs.size(); i++) {
                // Get tensor info
                auto tensor_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
                auto output_type = tensor_info.GetElementType();
                auto output_shape = tensor_info.GetShape();

                // Convert shape
                Shape shape;
                shape.dims.assign(output_shape.begin(), output_shape.end());
                
                // Create tensor with correct name and type
                Tensor output_tensor(onnx_output_names_[i], ConvertFromOnnxDataType(output_type), shape);
                
                // Get tensor data based on type
                switch (output_type) {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
                    const float* data = ort_outputs[i].GetTensorData<float>();
                    size_t element_count = shape.NumElements();
                    std::vector<float> tensor_data(data, data + element_count);
                    output_tensor.SetData(tensor_data);
                    break;
                }
                
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                    const int32_t* data = ort_outputs[i].GetTensorData<int32_t>();
                    size_t element_count = shape.NumElements();
                    std::vector<int32_t> tensor_data(data, data + element_count);
                    output_tensor.SetData(tensor_data);
                    break;
                }
                
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                    const int64_t* data = ort_outputs[i].GetTensorData<int64_t>();
                    size_t element_count = shape.NumElements();
                    std::vector<int64_t> tensor_data(data, data + element_count);
                    output_tensor.SetData(tensor_data);
                    break;
                }
                
                // Add other data types as needed
                
                default:
                    // For unsupported types, just create an empty tensor
                    std::cerr << "Unsupported output data type: " << output_type << std::endl;
                    break;
            
                }
                
                outputs.push_back(std::move(output_tensor));
            }   

            // Measure inference time
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
            
            // Update statistics
            stats_.inference_count++;
            stats_.last_inference_time_ns = duration;
            stats_.total_inference_time_ns += duration;
            
            std::cout << "ONNX inference completed in " << (duration / 1000000.0) << " ms" << std::endl;
            
            return true;
        } catch (const Ort::Exception& e) {
            last_error_ = std::string("ONNX Runtime inference error: ") + e.what();
            std::cerr << last_error_ << std::endl;
            return false;
        } catch (const std::exception& e) {
            last_error_ = std::string("ONNX inference error: ") + e.what();
            std::cerr << last_error_ << std::endl;
            return false;
        }
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
        try {
            std::cout << "Unloading ONNX model" << std::endl;
            
            // Release ONNX Runtime session
            if (onnx_session_) {
                onnx_session_.reset();
            }
            
            // Clear cached data
            onnx_input_names_.clear();
            onnx_output_names_.clear();
            onnx_input_shapes_.clear();
            onnx_output_shapes_.clear();
            onnx_input_types_.clear();
            onnx_output_types_.clear();
            
            std::cout << "ONNX model unloaded successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error unloading ONNX model: " << e.what() << std::endl;
        }
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