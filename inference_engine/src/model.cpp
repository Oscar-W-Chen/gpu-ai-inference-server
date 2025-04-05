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
#include <filesystem>
#include <cstring>        // For memcpy

namespace inference
{

    //====================================================
    // Tensor Implementation
    //====================================================

    /**
     * @brief Implementation details for the Tensor class
     * 
     * Handles data storage, memory management, and device transfer for tensors.
     * Supports both CPU and GPU memory with transparent synchronization.
     */
    class Tensor::TensorImpl
    {
    public:
        std::string name;
        DataType dtype;
        Shape shape;
        std::vector<uint8_t> cpu_data;
        void *gpu_data{nullptr};
        bool on_gpu{false};
        int device_id{0};

        /**
         * @brief Destructor that ensures GPU memory is freed
         */
        ~TensorImpl()
        {
            // Free GPU memory if allocated
            if (gpu_data != nullptr)
            {
                cudaFree(gpu_data);
                gpu_data = nullptr;
            }
        }

        /**
         * @brief Calculate the size in bytes for the tensor based on shape and data type
         * 
         * @return Size in bytes
         */
        size_t ByteSize() const
        {
            size_t element_size = 0;
            switch (dtype)
            {
            case DataType::FLOAT32:
                element_size = 4;
                break;
            case DataType::INT32:
                element_size = 4;
                break;
            case DataType::INT64:
                element_size = 8;
                break;
            case DataType::UINT8:
                element_size = 1;
                break;
            case DataType::INT8:
                element_size = 1;
                break;
            case DataType::BOOL:
                element_size = 1;
                break;
            case DataType::FP16:
                element_size = 2;
                break;
            case DataType::STRING:
                throw std::runtime_error("String tensors don't have fixed byte size");
            default:
                throw std::runtime_error("Unknown data type");
            }
            return shape.NumElements() * element_size;
        }

        /**
         * @brief Internal implementation for setting tensor data
         * 
         * @tparam T Data type of the input vector
         * @param data Vector containing the data to set
         * @param expected_type Expected data type for validation
         * @return true if successful, false otherwise
         */
        template <typename T>
        bool SetDataInternal(const std::vector<T> &data, DataType expected_type)
        {
            if (dtype != expected_type)
            {
                return false;
            }

            if (data.size() != shape.NumElements())
            {
                return false;
            }

            // Copy data to CPU buffer
            std::memcpy(cpu_data.data(), data.data(), ByteSize());

            // If on GPU, also update GPU memory
            if (on_gpu && gpu_data != nullptr)
            {
                cudaError_t error = cudaMemcpy(
                    gpu_data,
                    cpu_data.data(),
                    ByteSize(),
                    cudaMemcpyHostToDevice);

                if (error != cudaSuccess)
                {
                    return false;
                }
            }

            return true;
        }

        /**
         * @brief Internal implementation for getting tensor data
         * 
         * @tparam T Data type of the output vector
         * @param data Vector to store the retrieved data
         * @param expected_type Expected data type for validation
         * @return true if successful, false otherwise
         */
        template <typename T>
        bool GetDataInternal(std::vector<T> &data, DataType expected_type) const
        {
            if (dtype != expected_type)
            {
                return false;
            }

            // If on GPU, sync to CPU first
            if (on_gpu && gpu_data != nullptr)
            {
                cudaError_t error = cudaMemcpy(
                    const_cast<uint8_t *>(cpu_data.data()),
                    gpu_data,
                    ByteSize(),
                    cudaMemcpyDeviceToHost);

                if (error != cudaSuccess)
                {
                    return false;
                }
            }

            // Resize output vector
            data.resize(shape.NumElements());

            // Copy data to output vector
            std::memcpy(data.data(), cpu_data.data(), ByteSize());

            return true;
        }
    };

    /**
     * @brief Default constructor for Tensor
     */
    Tensor::Tensor() : impl_(new TensorImpl()) {}

    /**
     * @brief Parameterized constructor for Tensor
     * 
     * @param name The name of the tensor
     * @param dtype The data type of the tensor elements
     * @param shape The shape of the tensor
     */
    Tensor::Tensor(const std::string &name, DataType dtype, const Shape &shape)
        : impl_(new TensorImpl())
    {
        impl_->name = name;
        impl_->dtype = dtype;
        impl_->shape = shape;

        // Allocate CPU memory for the tensor
        size_t byte_size = impl_->ByteSize();
        impl_->cpu_data.resize(byte_size);
    }

    /**
     * @brief Copy constructor
     * 
     * @param other Tensor to copy from
     */
    Tensor::Tensor(const Tensor &other) : impl_(new TensorImpl(*other.impl_)) {}

    /**
     * @brief Move constructor
     * 
     * @param other Tensor to move from
     */
    Tensor::Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}

    /**
     * @brief Copy assignment operator
     * 
     * @param other Tensor to copy from
     * @return Reference to this tensor
     */
    Tensor &Tensor::operator=(const Tensor &other)
    {
        if (this != &other)
        {
            impl_ = std::make_unique<TensorImpl>(*other.impl_);
        }
        return *this;
    }

    /**
     * @brief Destructor
     */
    Tensor::~Tensor() = default;

    /**
     * @brief Get the tensor name
     * 
     * @return The tensor name
     */
    const std::string &Tensor::GetName() const
    {
        return impl_->name;
    }

    /**
     * @brief Get the tensor data type
     * 
     * @return The tensor data type
     */
    DataType Tensor::GetDataType() const
    {
        return impl_->dtype;
    }

    /**
     * @brief Get the tensor shape
     * 
     * @return The tensor shape
     */
    const Shape &Tensor::GetShape() const
    {
        return impl_->shape;
    }

    /**
     * @brief Reshape the tensor to a new shape
     * 
     * @param new_shape The new shape to resize to
     * @return true if successful, false otherwise
     */
    bool Tensor::Reshape(const Shape &new_shape)
    {
        size_t old_elements = impl_->shape.NumElements();
        size_t new_elements = 1;
        for (auto dim : new_shape.dims)
        {
            new_elements *= dim;
        }
        if (old_elements != new_elements)
        {
            // Need to resize the data buffer
            impl_->shape = new_shape;

            // Resize CPU data
            impl_->cpu_data.resize(impl_->ByteSize());

            // If on GPU, need to reallocate
            if (impl_->on_gpu && impl_->gpu_data != nullptr)
            {
                cudaFree(impl_->gpu_data);
                impl_->gpu_data = nullptr;

                // Allocate new GPU memory
                cudaError_t error = cudaMalloc(&impl_->gpu_data, impl_->ByteSize());
                if (error != cudaSuccess)
                {
                    impl_->on_gpu = false;
                    return false;
                }
            }
            else
            {
                // Can just update the shape
                impl_->shape = new_shape;
            }
        }
        return true;
    }

    /**
     * @brief Move tensor data to GPU memory
     * 
     * @param device_id GPU device ID to use
     * @return true if successful, false otherwise
     */
    bool Tensor::toGPU(int device_id)
    {
        if (!cuda::IsCudaAvailable())
        {
            return false;
        }

        impl_->device_id = device_id;

        // Set device
        cudaError_t error = cudaSetDevice(device_id);
        if (error != cudaSuccess)
        {
            return false;
        }

        // Already on GPU?
        if (impl_->on_gpu && impl_->gpu_data != nullptr)
        {
            return true;
        }

        // Allocate GPU memory
        error = cudaMalloc(&impl_->gpu_data, impl_->ByteSize());
        if (error != cudaSuccess)
        {
            return false;
        }

        // Copy data to GPU
        error = cudaMemcpy(
            impl_->gpu_data,
            impl_->cpu_data.data(),
            impl_->ByteSize(),
            cudaMemcpyHostToDevice);

        if (error != cudaSuccess)
        {
            cudaFree(impl_->gpu_data);
            impl_->gpu_data = nullptr;
            return false;
        }

        impl_->on_gpu = true;
        return true;
    }

    /**
     * @brief Move tensor data back to CPU memory
     * 
     * @return true if successful, false otherwise
     */
    bool Tensor::toCPU()
    {
        if (!impl_->on_gpu || impl_->gpu_data == nullptr)
        {
            return true; // Already on CPU
        }

        // Set device
        cudaError_t error = cudaSetDevice(impl_->device_id);
        if (error != cudaSuccess)
        {
            return false;
        }

        // Copy data from GPU to CPU
        error = cudaMemcpy(
            impl_->cpu_data.data(),
            impl_->gpu_data,
            impl_->ByteSize(),
            cudaMemcpyDeviceToHost);

        if (error != cudaSuccess)
        {
            return false;
        }

        // Free GPU memory
        cudaFree(impl_->gpu_data);
        impl_->gpu_data = nullptr;
        impl_->on_gpu = false;

        return true;
    }

    // Template specializations for SetData
    template <>
    bool Tensor::SetData(const std::vector<float> &data)
    {
        return impl_->SetDataInternal(data, DataType::FLOAT32);
    }

    template <>
    bool Tensor::GetData(std::vector<float> &data) const
    {
        return impl_->GetDataInternal(data, DataType::FLOAT32);
    }

    template <>
    bool Tensor::SetData(const std::vector<int> &data)
    {
        return impl_->SetDataInternal(data, DataType::INT32);
    }

    template <>
    bool Tensor::GetData(std::vector<int> &data) const
    {
        return impl_->GetDataInternal(data, DataType::INT32);
    }

    template <>
    bool Tensor::SetData(const std::vector<long> &data)
    {
        return impl_->SetDataInternal(data, DataType::INT64);
    }

    template <>
    bool Tensor::GetData(std::vector<long> &data) const
    {
        return impl_->GetDataInternal(data, DataType::INT64);
    }

    //====================================================
    // Model Implementation
    //====================================================

    /**
     * @brief Internal implementation for the Model class
     * 
     * Handles all the details of model loading, unloading, and inference
     * for different model types (ONNX, TensorRT, etc.)
     */
    class ModelImpl
    {
    public:
        /**
         * @brief Constructor for ModelImpl
         * 
         * @param model_path Path to the model file or directory
         * @param type Model type (ONNX, TensorRT, etc.)
         * @param config Model configuration
         * @param device Device type (CPU/GPU)
         * @param device_id Device ID if using GPU
         */
        ModelImpl(const std::string &model_path,
                  ModelType type,
                  const ModelConfig &config,
                  DeviceType device,
                  int device_id)
            : model_path_(model_path),
              type_(type),
              config_(config),
              device_type_(device),
              device_id_(device_id),
              loaded_(false)
        {
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

        /**
         * @brief Destructor ensures model is unloaded
         */
        ~ModelImpl()
        {
            if (loaded_)
            {
                Unload();
            }
        }

        /**
         * @brief Load the model into memory
         * 
         * @return true if successful, false otherwise
         */
        bool Load()
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Check model file exists
            if (!FileExists(model_path_))
            {
                last_error_ = "Model file not found: " + model_path_;
                return false;
            }

            // Model type specific loading
            switch (type_)
            {
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
                                         end_time - start_time)
                                         .count();

            return loaded_;
        }

        /**
         * @brief Run inference using the loaded model
         * 
         * @param inputs Vector of input tensors
         * @param outputs Vector to store output tensors
         * @return true if successful, false otherwise
         */
        bool Infer(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            if (!loaded_)
            {
                last_error_ = "Model not loaded";
                return false;
            }

            // Validate inputs
            if (!ValidateInputs(inputs))
            {
                return false;
            }

            // Model-specific inference
            auto start_time = std::chrono::high_resolution_clock::now();

            bool success = false;
            switch (type_)
            {
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
                                end_time - start_time)
                                .count();

            // Update stats
            stats_.inference_count++;
            stats_.total_inference_time_ns += duration;
            stats_.last_inference_time_ns = duration;

            return success;
        }

        /**
         * @brief Unload the model from memory
         */
        void Unload()
        {
            // Model-specific unloading
            switch (type_)
            {
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

        /**
         * @brief Get metadata about the model
         * 
         * @return ModelMetadata struct with model information
         */
        ModelMetadata GetMetadata() const
        {
            return metadata_;
        }

        /**
         * @brief Check if model is currently loaded
         * 
         * @return true if loaded, false otherwise
         */
        bool IsLoaded() const
        {
            return loaded_;
        }

        /**
         * @brief Get the last error message
         * 
         * @return Error message string
         */
        std::string GetLastError() const
        {
            return last_error_;
        }

        /**
         * @brief Get model statistics
         * 
         * @return Stats struct with performance metrics
         */
        Model::Stats GetStats() const
        {
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

        // ONNX Runtime specific members
        Ort::Env onnx_env_;
        std::unique_ptr<Ort::Session> onnx_session_;
        std::vector<std::string> onnx_input_names_;
        std::vector<std::string> onnx_output_names_;
        std::vector<std::vector<int64_t>> onnx_input_shapes_;
        std::vector<std::vector<int64_t>> onnx_output_shapes_;
        std::vector<ONNXTensorElementDataType> onnx_input_types_;
        std::vector<ONNXTensorElementDataType> onnx_output_types_;

        /**
         * @brief Check if a file exists
         * 
         * @param path Path to check
         * @return true if exists, false otherwise
         */
        bool FileExists(const std::string &path)
        {
            std::ifstream file(path);
            return file.good();
        }

        /**
         * @brief Validate input tensors against model requirements
         * 
         * @param inputs Vector of input tensors to validate
         * @return true if valid, false otherwise
         */
        bool ValidateInputs(const std::vector<Tensor> &inputs)
        {
            // Check input count
            if (inputs.size() != config_.input_names.size())
            {
                last_error_ = "Expected " + std::to_string(config_.input_names.size()) +
                              " inputs, got " + std::to_string(inputs.size());
                return false;
            }

            for (const auto &input : inputs)
            {
                // Check name
                auto it = std::find(config_.input_names.begin(),
                                    config_.input_names.end(),
                                    input.GetName());
                if (it == config_.input_names.end())
                {
                    last_error_ = "Unexpected input name: " + input.GetName();
                    return false;
                }

                // Check data type
                if (config_.input_types.count(input.GetName()) > 0 &&
                    config_.input_types.at(input.GetName()) != input.GetDataType())
                {
                    last_error_ = "Input data type mismatch for " + input.GetName();
                    return false;
                }

                // Check shape (if not dynamic)
                if (config_.input_shapes.count(input.GetName()) > 0)
                {
                    const auto &expected_shape = config_.input_shapes.at(input.GetName());
                    const auto &actual_shape = input.GetShape();

                    // Check dimension count
                    if (expected_shape.dims.size() != actual_shape.dims.size())
                    {
                        last_error_ = "Input shape mismatch for " + input.GetName() +
                                      ": expected " + std::to_string(expected_shape.dims.size()) +
                                      " dimensions, got " + std::to_string(actual_shape.dims.size());
                        return false;
                    }

                    // Check each dimension (allowing for dynamic dimensions with -1)
                    for (size_t i = 0; i < expected_shape.dims.size(); i++)
                    {
                        if (expected_shape.dims[i] != -1 &&
                            expected_shape.dims[i] != actual_shape.dims[i])
                        {
                            last_error_ = "Input shape mismatch for " + input.GetName() +
                                          " at dimension " + std::to_string(i);
                            return false;
                        }
                    }
                }
            }

            return true;
        }

        /**
         * @brief Load a TensorFlow model
         * 
         * @return true if successful, false otherwise
         */
        bool LoadTensorFlowModel()
        {
            // Implementation placeholder
            last_error_ = "TensorFlow model loading not implemented";
            return false;
        }

        /**
         * @brief Load a TensorRT model
         * 
         * @return true if successful, false otherwise
         */
        bool LoadTensorRTModel()
        {
            // Implementation placeholder
            last_error_ = "TensorRT model loading not implemented";
            return false;
        }

        /**
         * @brief Load an ONNX model into memory and prepare it for inference
         * 
         * @return true if successful, false otherwise
         */
        bool LoadONNXModel()
        {
            try
            {
                // Construct the full path to the ONNX file
                std::string onnx_file_path = model_path_ + "/model.onnx";

                // Verify the model file exists
                if (!std::filesystem::exists(onnx_file_path))
                {
                    last_error_ = "ONNX model file not found: " + onnx_file_path;
                    return false;
                }

                // Create ONNX Runtime environment with appropriate logging level
                onnx_env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "inference-server");

                // Configure session options
                Ort::SessionOptions session_options;
                ConfigureSessionOptions(session_options);

                // Create ONNX Runtime session with the FULL file path
                onnx_session_ = std::make_unique<Ort::Session>(onnx_env_, onnx_file_path.c_str(), session_options);

                // Extract model metadata and update internal structures
                if (!ExtractModelMetadata())
                {
                    return false;
                }

                // Update memory usage statistics
                stats_.memory_usage_bytes = EstimateModelMemoryUsage();

                loaded_ = true;
                return true;
            }
            catch (const Ort::Exception &e)
            {
                last_error_ = std::string("ONNX Runtime error: ") + e.what();
                return false;
            }
            catch (const std::exception &e)
            {
                last_error_ = std::string("ONNX model loading error: ") + e.what();
                return false;
            }
        }

        /**
         * @brief Configure ONNX Runtime session options
         * 
         * @param session_options SessionOptions object to configure
         */
        void ConfigureSessionOptions(Ort::SessionOptions &session_options)
        {
            // Enable GPU execution if available and requested
            if (device_type_ == DeviceType::GPU && cuda::IsCudaAvailable())
            {
                // Configure CUDA provider options
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = device_id_;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
                cuda_options.gpu_mem_limit = 0;         // No limit
                cuda_options.arena_extend_strategy = 0; // Default strategy
                cuda_options.do_copy_in_default_stream = 1;

                // Register CUDA execution provider
                session_options.AppendExecutionProvider_CUDA(cuda_options);
            }

            // Set graph optimization level (enable all optimizations)
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            // Set thread count for CPU operations
            session_options.SetIntraOpNumThreads(1);

            // Set execution mode (sequential or parallel)
            session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        }

        /**
         * @brief Extract model metadata from ONNX Runtime session
         * 
         * @return true if successful, false otherwise
         */
        bool ExtractModelMetadata()
        {
            Ort::AllocatorWithDefaultOptions allocator;

            // Get number of inputs and outputs
            size_t num_inputs = onnx_session_->GetInputCount();
            size_t num_outputs = onnx_session_->GetOutputCount();

            // Clear previous data
            onnx_input_names_.clear();
            onnx_output_names_.clear();
            onnx_input_shapes_.clear();
            onnx_output_shapes_.clear();
            onnx_input_types_.clear();
            onnx_output_types_.clear();

            // Reserve space for vectors
            onnx_input_names_.resize(num_inputs);
            onnx_input_shapes_.resize(num_inputs);
            onnx_input_types_.resize(num_inputs);

            onnx_output_names_.resize(num_outputs);
            onnx_output_shapes_.resize(num_outputs);
            onnx_output_types_.resize(num_outputs);

            // Process input metadata
            for (size_t i = 0; i < num_inputs; i++)
            {
                // Get input name
                auto input_name = onnx_session_->GetInputNameAllocated(i, allocator);
                onnx_input_names_[i] = input_name.get();

                // Get input type info
                Ort::TypeInfo type_info = onnx_session_->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                // Extract data type
                onnx_input_types_[i] = tensor_info.GetElementType();

                // Extract shape information
                onnx_input_shapes_[i] = tensor_info.GetShape();
            }

            // Process output metadata
            for (size_t i = 0; i < num_outputs; i++)
            {
                // Get output name
                auto output_name = onnx_session_->GetOutputNameAllocated(i, allocator);
                onnx_output_names_[i] = output_name.get();

                // Get output type info
                Ort::TypeInfo type_info = onnx_session_->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

                // Extract data type
                onnx_output_types_[i] = tensor_info.GetElementType();

                // Extract shape information
                onnx_output_shapes_[i] = tensor_info.GetShape();
            }

            return true;
        }

        /**
         * @brief Estimate memory usage for the ONNX model
         * 
         * @return Estimated memory usage in bytes
         */
        size_t EstimateModelMemoryUsage() const
        {
            size_t total_size = 0;

            // Calculate element size based on ONNX data type
            auto getElementSize = [](ONNXTensorElementDataType type) -> size_t
            {
                switch (type)
                {
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    return 4; // 4 bytes
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                    return 8; // 8 bytes
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                    return 1; // 1 byte
                case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                    return 2; // 2 bytes
                default:
                    return 4; // Default to 4 bytes
                }
            };

            // Calculate tensor size
            auto calculateTensorSize = [&getElementSize](const std::vector<int64_t> &shape,
                                                         ONNXTensorElementDataType type) -> size_t
            {
                size_t num_elements = 1;
                for (const auto &dim : shape)
                {
                    if (dim > 0)
                    { // Skip dynamic dimensions (-1)
                        num_elements *= dim;
                    }
                }
                return num_elements * getElementSize(type);
            };

            // Sum up input tensor sizes
            for (size_t i = 0; i < onnx_input_shapes_.size(); i++)
            {
                total_size += calculateTensorSize(onnx_input_shapes_[i], onnx_input_types_[i]);
            }

            // Sum up output tensor sizes
            for (size_t i = 0; i < onnx_output_shapes_.size(); i++)
            {
                total_size += calculateTensorSize(onnx_output_shapes_[i], onnx_output_types_[i]);
            }

            // Add base memory overhead for model weights and runtime
            constexpr size_t BASE_OVERHEAD_BYTES = 10 * 1024 * 1024; // 10 MB baseline

            return total_size + BASE_OVERHEAD_BYTES;
        }

        /**
         * @brief Convert from ONNX data type to inference::DataType
         * 
         * @param onnx_type ONNX data type
         * @return Corresponding inference::DataType
         */
        DataType ConvertFromOnnxDataType(ONNXTensorElementDataType onnx_type)
        {
            switch (onnx_type)
            {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                return DataType::FLOAT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                return DataType::INT32;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                return DataType::INT64;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
                return DataType::UINT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                return DataType::INT8;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
                return DataType::STRING;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return DataType::BOOL;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                return DataType::FP16;
            default:
                return DataType::UNKNOWN;
            }
        }

        /**
         * @brief Convert from inference::DataType to ONNX data type
         * 
         * @param dtype inference::DataType
         * @return Corresponding ONNX data type
         */
        ONNXTensorElementDataType ConvertToOnnxDataType(DataType dtype)
        {
            switch (dtype)
            {
            case DataType::FLOAT32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
            case DataType::INT32:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
            case DataType::INT64:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
            case DataType::UINT8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
            case DataType::INT8:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
            case DataType::STRING:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
            case DataType::BOOL:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
            case DataType::FP16:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            default:
                return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            }
        }

        /**
         * @brief Load a PyTorch model
         * 
         * @return true if successful, false otherwise
         */
        bool LoadPyTorchModel()
        {
            // Implementation placeholder
            last_error_ = "PyTorch model loading not implemented";
            return false;
        }

        /**
         * @brief Load a custom model
         * 
         * @return true if successful, false otherwise
         */
        bool LoadCustomModel()
        {
            // Implementation placeholder
            last_error_ = "Custom model loading not implemented";
            return false;
        }

        /**
         * @brief Run inference with TensorFlow model
         * 
         * @param inputs Input tensors
         * @param outputs Output tensors (will be filled)
         * @return true if successful, false otherwise
         */
        bool InferTensorFlow(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            // Implementation placeholder
            last_error_ = "TensorFlow inference not implemented";
            return false;
        }

        /**
         * @brief Run inference with TensorRT model
         * 
         * @param inputs Input tensors
         * @param outputs Output tensors (will be filled)
         * @return true if successful, false otherwise
         */
        bool InferTensorRT(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            // Implementation placeholder
            last_error_ = "TensorRT inference not implemented";
            return false;
        }

        /**
         * @brief Execute inference using the loaded ONNX model
         * 
         * @param inputs Vector of input tensors matching the model's expected inputs
         * @param outputs Vector that will be filled with output tensors
         * @return true if successful, false otherwise
         */
        bool InferONNX(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            if (!onnx_session_)
            {
                last_error_ = "ONNX model is not loaded";
                return false;
            }

            try
            {
                // Create memory info for CPU allocations
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator,
                    OrtMemType::OrtMemTypeDefault);

                // Prepare input/output name mappings
                std::unordered_map<std::string, size_t> input_name_to_index;
                for (size_t i = 0; i < onnx_input_names_.size(); i++)
                {
                    input_name_to_index[onnx_input_names_[i]] = i;
                }

                // Check that all required inputs are provided
                for (const auto &input_name : onnx_input_names_)
                {
                    if (std::none_of(inputs.begin(), inputs.end(),
                                     [&input_name](const Tensor &t)
                                     { return t.GetName() == input_name; }))
                    {
                        last_error_ = "Required input tensor not provided: " + input_name;
                        return false;
                    }
                }

                // Create C-style input and output name arrays for ONNX Runtime
                std::vector<const char *> input_names_cstr;
                std::vector<const char *> output_names_cstr;
                input_names_cstr.reserve(onnx_input_names_.size());
                output_names_cstr.reserve(onnx_output_names_.size());

                for (const auto &name : onnx_input_names_)
                {
                    input_names_cstr.push_back(name.c_str());
                }
                for (const auto &name : onnx_output_names_)
                {
                    output_names_cstr.push_back(name.c_str());
                }

                // Create ONNX input tensors in the order expected by the model
                std::vector<Ort::Value> ort_inputs;
                ort_inputs.resize(onnx_input_names_.size());

                // Convert each input tensor to ONNX format
                for (const auto &input : inputs)
                {
                    auto it = input_name_to_index.find(input.GetName());
                    if (it == input_name_to_index.end())
                    {
                        continue; // Skip inputs not required by the model
                    }

                    size_t input_idx = it->second;
                    const auto &shape = input.GetShape();
                    std::vector<int64_t> input_shape(shape.dims.begin(), shape.dims.end());

                    // Convert based on data type
                    switch (input.GetDataType())
                    {
                    case DataType::FLOAT32:
                    {
                        std::vector<float> data;
                        if (!input.GetData(data) || data.size() != shape.NumElements())
                        {
                            last_error_ = "Invalid FLOAT32 data for input: " + input.GetName();
                            return false;
                        }

                        ort_inputs[input_idx] = Ort::Value::CreateTensor<float>(
                            memory_info, data.data(), data.size(),
                            input_shape.data(), input_shape.size());
                        break;
                    }

                    case DataType::INT32:
                    {
                        std::vector<int32_t> data;
                        if (!input.GetData(data) || data.size() != shape.NumElements())
                        {
                            last_error_ = "Invalid INT32 data for input: " + input.GetName();
                            return false;
                        }

                        ort_inputs[input_idx] = Ort::Value::CreateTensor<int32_t>(
                            memory_info, data.data(), data.size(),
                            input_shape.data(), input_shape.size());
                        break;
                    }

                    default:
                        last_error_ = "Unsupported data type for input: " + input.GetName();
                        return false;
                    }
                }

                // Run inference
                auto ort_outputs = onnx_session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names_cstr.data(),
                    ort_inputs.data(),
                    ort_inputs.size(),
                    output_names_cstr.data(),
                    output_names_cstr.size());

                // Convert ONNX outputs to our tensor format
                outputs.clear();
                outputs.reserve(ort_outputs.size());

                for (size_t i = 0; i < ort_outputs.size(); i++)
                {
                    auto tensor_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
                    auto output_type = tensor_info.GetElementType();
                    auto output_shape = tensor_info.GetShape();

                    // Convert shape
                    Shape shape;
                    shape.dims.assign(output_shape.begin(), output_shape.end());

                    // Create output tensor
                    Tensor output_tensor(onnx_output_names_[i], ConvertFromOnnxDataType(output_type), shape);

                    // Copy data based on type
                    switch (output_type)
                    {
                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                    {
                        const float *data = ort_outputs[i].GetTensorData<float>();
                        std::vector<float> tensor_data(data, data + shape.NumElements());
                        output_tensor.SetData(tensor_data);
                        break;
                    }

                    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                    {
                        const int32_t *data = ort_outputs[i].GetTensorData<int32_t>();
                        std::vector<int32_t> tensor_data(data, data + shape.NumElements());
                        output_tensor.SetData(tensor_data);
                        break;
                    }

                    default:
                        // Log but continue with other tensors
                        break;
                    }

                    outputs.push_back(std::move(output_tensor));
                }

                return true;
            }
            catch (const Ort::Exception &e)
            {
                last_error_ = std::string("ONNX Runtime inference error: ") + e.what();
                return false;
            }
            catch (const std::exception &e)
            {
                last_error_ = std::string("ONNX inference error: ") + e.what();
                return false;
            }
        }

        /**
         * @brief Run inference with PyTorch model
         * 
         * @param inputs Input tensors
         * @param outputs Output tensors (will be filled)
         * @return true if successful, false otherwise
         */
        bool InferPyTorch(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            // Implementation placeholder
            last_error_ = "PyTorch inference not implemented";
            return false;
        }

        /**
         * @brief Run inference with custom model
         * 
         * @param inputs Input tensors
         * @param outputs Output tensors (will be filled)
         * @return true if successful, false otherwise
         */
        bool InferCustom(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
        {
            // Implementation placeholder
            last_error_ = "Custom inference not implemented";
            return false;
        }

        /**
         * @brief Unload a TensorFlow model
         */
        void UnloadTensorFlow()
        {
            // Implementation placeholder
        }

        /**
         * @brief Unload a TensorRT model
         */
        void UnloadTensorRT()
        {
            // Implementation placeholder
        }

        /**
         * @brief Unload an ONNX model
         */
        void UnloadONNX()
        {
            try
            {
                // Release ONNX Runtime session
                if (onnx_session_)
                {
                    onnx_session_.reset();
                }

                // Clear cached data
                onnx_input_names_.clear();
                onnx_output_names_.clear();
                onnx_input_shapes_.clear();
                onnx_output_shapes_.clear();
                onnx_input_types_.clear();
                onnx_output_types_.clear();
            }
            catch (const std::exception &e)
            {
                // Log but continue
            }
        }

        /**
         * @brief Unload a PyTorch model
         */
        void UnloadPyTorch()
        {
            // Implementation placeholder
        }

        /**
         * @brief Unload a custom model
         */
        void UnloadCustom()
        {
            // Implementation placeholder
        }
    };

    //====================================================
    // Model Public Methods
    //====================================================

    /**
     * @brief Construct a new Model object
     * 
     * @param model_path Path to the model file or directory
     * @param type Model type (ONNX, TensorRT, etc.)
     * @param config Model configuration
     * @param device Device type (CPU/GPU)
     * @param device_id Device ID if using GPU
     */
    Model::Model(const std::string &model_path,
                 ModelType type,
                 const ModelConfig &config,
                 DeviceType device,
                 int device_id)
        : impl_(new ModelImpl(model_path, type, config, device, device_id))
    {
    }

    /**
     * @brief Destructor
     */
    Model::~Model() = default;

    /**
     * @brief Move constructor
     */
    Model::Model(Model &&other) noexcept = default;

    /**
     * @brief Move assignment operator
     */
    Model &Model::operator=(Model &&other) noexcept = default;

    /**
     * @brief Load the model into memory
     * 
     * @return true if successful, false otherwise
     */
    bool Model::Load()
    {
        return impl_->Load();
    }

    /**
     * @brief Run inference using this model
     * 
     * @param inputs Vector of input tensors
     * @param outputs Vector to store output tensors
     * @return true if successful, false otherwise
     */
    bool Model::Infer(const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs)
    {
        return impl_->Infer(inputs, outputs);
    }

    /**
     * @brief Get model metadata
     * 
     * @return ModelMetadata structure
     */
    ModelMetadata Model::GetMetadata() const
    {
        return impl_->GetMetadata();
    }

    /**
     * @brief Check if model is loaded
     * 
     * @return true if loaded, false otherwise
     */
    bool Model::IsLoaded() const
    {
        return impl_->IsLoaded();
    }

    /**
     * @brief Unload the model from memory
     */
    void Model::Unload()
    {
        impl_->Unload();
    }

    /**
     * @brief Get the last error message
     * 
     * @return Error message string
     */
    std::string Model::GetLastError() const
    {
        return impl_->GetLastError();
    }

    /**
     * @brief Get model statistics
     * 
     * @return Stats structure with performance information
     */
    Model::Stats Model::GetStats() const
    {
        return impl_->GetStats();
    }

} // namespace inference