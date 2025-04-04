// Bridge implementation between C and C++ for the inference server
#include "inference_bridge.h"
#include "model.h"
#include "cuda_utils.h"
#include "model_repository.h"
#include <fstream>
#include <iomanip> // for std::setw, std::hex, etc.
#include <sstream> // for std::stringstream
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <cstring>

// Bridge class for InferenceManager
// Bridge class for InferenceManager
struct InferenceManager_t
{
    std::string model_repository_path;
    std::unique_ptr<inference::ModelRepository> repository;
    std::unordered_map<std::string, std::unique_ptr<inference::Model>> models;
};

// Bridge class for Model
struct Model_t
{
    std::unique_ptr<inference::Model> owned_model;
    inference::Model* model;  // May point to owned_model.get() or an external model

    // Constructor for owned model
    Model_t(std::unique_ptr<inference::Model> m) : owned_model(std::move(m)), model(owned_model.get()) {}
    
    // Constructor for non-owned model reference
    Model_t(std::nullptr_t) : owned_model(nullptr), model(nullptr) {}
    
    // Destructor only deletes if we own the model
    ~Model_t() {
        // owned_model's destructor will handle deletion if needed
    }
};

// Bridge class for Tensor
struct Tensor_t
{
    inference::Tensor tensor; // Direct object rather than pointer

    // Constructor with direct parameters
    Tensor_t(const std::string &name, inference::DataType dtype, const inference::Shape &shape)
        : tensor(name, dtype, shape) {}
};

// Helper function to allocate C string copy
char *strdup_helper(const std::string &str)
{
    char *result = strdup(str.c_str());
    return result;
}

// Helper function to convert between C and C++ enums
inference::ModelType convert_model_type(ModelType type)
{
    switch (type)
    {
    case MODEL_TENSORFLOW:
        return inference::ModelType::TENSORFLOW;
    case MODEL_TENSORRT:
        return inference::ModelType::TENSORRT;
    case MODEL_ONNX:
        return inference::ModelType::ONNX;
    case MODEL_PYTORCH:
        return inference::ModelType::PYTORCH;
    case MODEL_CUSTOM:
        return inference::ModelType::CUSTOM;
    default:
        return inference::ModelType::UNKNOWN;
    }
}

inference::DeviceType convert_device_type(DeviceType device)
{
    switch (device)
    {
    case DEVICE_GPU:
        return inference::DeviceType::GPU;
    case DEVICE_CPU:
    default:
        return inference::DeviceType::CPU;
    }
}

inference::DataType convert_data_type(DataType dtype)
{
    switch (dtype)
    {
    case DATATYPE_FLOAT32:
        return inference::DataType::FLOAT32;
    case DATATYPE_INT32:
        return inference::DataType::INT32;
    case DATATYPE_INT64:
        return inference::DataType::INT64;
    case DATATYPE_UINT8:
        return inference::DataType::UINT8;
    case DATATYPE_INT8:
        return inference::DataType::INT8;
    case DATATYPE_STRING:
        return inference::DataType::STRING;
    case DATATYPE_BOOL:
        return inference::DataType::BOOL;
    case DATATYPE_FP16:
        return inference::DataType::FP16;
    default:
        return inference::DataType::UNKNOWN;
    }
}

// Convert C++ ModelType to C ModelType
ModelType convert_to_c_model_type(inference::ModelType type)
{
    switch (type)
    {
    case inference::ModelType::TENSORFLOW:
        return MODEL_TENSORFLOW;
    case inference::ModelType::TENSORRT:
        return MODEL_TENSORRT;
    case inference::ModelType::ONNX:
        return MODEL_ONNX;
    case inference::ModelType::PYTORCH:
        return MODEL_PYTORCH;
    case inference::ModelType::CUSTOM:
        return MODEL_CUSTOM;
    default:
        return MODEL_UNKNOWN;
    }
}

// Convert C++ DataType to C DataType
DataType convert_to_c_data_type(inference::DataType dtype)
{
    switch (dtype)
    {
    case inference::DataType::FLOAT32:
        return DATATYPE_FLOAT32;
    case inference::DataType::INT32:
        return DATATYPE_INT32;
    case inference::DataType::INT64:
        return DATATYPE_INT64;
    case inference::DataType::UINT8:
        return DATATYPE_UINT8;
    case inference::DataType::INT8:
        return DATATYPE_INT8;
    case inference::DataType::STRING:
        return DATATYPE_STRING;
    case inference::DataType::BOOL:
        return DATATYPE_BOOL;
    case inference::DataType::FP16:
        return DATATYPE_FP16;
    default:
        return DATATYPE_UNKNOWN;
    }
}

// CUDA utility functions
extern "C"
{
    bool IsCudaAvailable()
    {
        return inference::cuda::IsCudaAvailable();
    }

    int GetDeviceCount()
    {
        return inference::cuda::GetDeviceCount();
    }

    const char *GetDeviceInfo(int device_id)
    {
        std::string info = inference::cuda::GetDeviceInfo(device_id);
        return strdup_helper(info);
    }

    CudaMemoryInfo GetMemoryInfo(int device_id)
    {
        // Call the C++ function from cuda_utils
        inference::cuda::MemoryInfo memInfo = inference::cuda::GetMemoryInfo(device_id);

        // Convert to C struct for the bridge
        CudaMemoryInfo cInfo;
        cInfo.total = memInfo.total;
        cInfo.free = memInfo.free;
        cInfo.used = memInfo.used;

        return cInfo;
    }

    // Inference Manager Functions
    InferenceManagerHandle InferenceInitialize(const char *model_repository_path)
    {
        try
        {
            InferenceManager_t *manager = new InferenceManager_t();
            manager->model_repository_path = model_repository_path ? model_repository_path : "";

            // Initialize model repository
            manager->repository = std::make_unique<inference::ModelRepository>(manager->model_repository_path);
            if (manager->repository)
            {
                manager->repository->ScanRepository();
            }

            return manager;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in InferenceInitialize: " << e.what() << std::endl;
            return nullptr;
        }
    }

    void InferenceShutdown(InferenceManagerHandle handle)
    {
        delete handle;
    }

    bool InferenceLoadModel(InferenceManagerHandle handle, const char *model_name, const char *version, ErrorMessage *error)
    {
        if (!handle || !model_name)
        {
            if (error)
                *error = strdup_helper("Invalid handle or model name");
            return false;
        }

        try
        {
            std::cerr << "Debug [InferenceLoadModel]: Loading model: " << model_name
                      << ", version: " << (version ? version : "latest")
                      << ", repo path: " << handle->model_repository_path << std::endl;

            // In a real implementation, we would load the model from the repository
            if (!handle->repository)
            {
                std::cerr << "Debug [InferenceLoadModel]: Repository is null" << std::endl;
                if (error)
                    *error = strdup_helper("Repository is not initialized");
                return false;
            }

            // Get model path and configuration from repository
            std::string resolved_version = version ? version : handle->repository->GetLatestVersion(model_name);
            std::string model_path = handle->repository->GetModelPath(model_name, resolved_version);

            std::cerr << "Debug [InferenceLoadModel]: Resolved version: " << resolved_version << std::endl;
            std::cerr << "Debug [InferenceLoadModel]: Resolved model path: " << model_path << std::endl;

            // Verify that the model directory exists
            if (!std::filesystem::exists(model_path))
            {
                std::cerr << "Debug [InferenceLoadModel]: Model path does not exist: " << model_path << std::endl;
                if (error)
                    *error = strdup_helper(("Model path not found: " + model_path).c_str());
                return false;
            }

            // Check if model is already loaded
            if (handle->models.find(model_name) != handle->models.end())
            {
                std::cerr << "Debug [InferenceLoadModel]: Model already loaded: " << model_name << std::endl;
                if (error)
                    *error = strdup_helper("Model already loaded");
                return false;
            }

            // Check for model.onnx file
            std::string onnx_file_path = model_path + "/model.onnx";
            if (std::filesystem::exists(onnx_file_path))
            {
                std::cerr << "Debug [InferenceLoadModel]: Found ONNX file at: " << onnx_file_path << std::endl;
                std::cerr << "Debug [InferenceLoadModel]: File size: "
                          << std::filesystem::file_size(onnx_file_path) << " bytes" << std::endl;

                // Try to read a few bytes to check if file is accessible
                std::ifstream file(onnx_file_path, std::ios::binary);
                if (file.is_open())
                {
                    char header[16];
                    file.read(header, sizeof(header));
                    std::cerr << "Debug [InferenceLoadModel]: Successfully read "
                              << file.gcount() << " bytes from file" << std::endl;

                    // Print first few bytes in hex for debugging
                    std::stringstream ss;
                    ss << "Debug [InferenceLoadModel]: File header bytes: ";
                    for (int i = 0; i < file.gcount() && i < 16; i++)
                    {
                        ss << std::hex << std::setw(2) << std::setfill('0')
                           << (int)(unsigned char)header[i] << " ";
                    }
                    std::cerr << ss.str() << std::endl;
                }
                else
                {
                    std::cerr << "Debug [InferenceLoadModel]: Failed to open file for reading" << std::endl;
                    if (error)
                        *error = strdup_helper(("Cannot read model file: " + onnx_file_path).c_str());
                    return false;
                }
            }
            else
            {
                std::cerr << "Debug [InferenceLoadModel]: ONNX file not found at: " << onnx_file_path << std::endl;

                // List directory contents to see what's there
                std::cerr << "Debug [InferenceLoadModel]: Directory contents of " << model_path << ":" << std::endl;
                for (const auto &entry : std::filesystem::directory_iterator(model_path))
                {
                    std::cerr << "  - " << entry.path().filename().string();
                    if (entry.is_directory())
                    {
                        std::cerr << " (directory)";
                    }
                    else
                    {
                        std::cerr << " (" << std::filesystem::file_size(entry.path()) << " bytes)";
                    }
                    std::cerr << std::endl;
                }
            }

            // Load the model configuration from repository
            inference::ModelConfig config = handle->repository->GetModelConfig(model_name, resolved_version);
            if (config.type == inference::ModelType::UNKNOWN)
            {
                std::cerr << "Debug [InferenceLoadModel]: Unable to determine model type for: "
                          << model_name << std::endl;
                if (error)
                    *error = strdup_helper("Unable to determine model type");
                return false;
            }

            std::cerr << "Debug [InferenceLoadModel]: Model type determined: "
                      << static_cast<int>(config.type) << std::endl;
            std::cerr << "Debug [InferenceLoadModel]: Input names ("
                      << config.input_names.size() << "): ";
            for (const auto &name : config.input_names)
            {
                std::cerr << name << " ";
            }
            std::cerr << std::endl;

            std::cerr << "Debug [InferenceLoadModel]: Output names ("
                      << config.output_names.size() << "): ";
            for (const auto &name : config.output_names)
            {
                std::cerr << name << " ";
            }
            std::cerr << std::endl;

            // Create and load the model
            std::cerr << "Debug [InferenceLoadModel]: Creating model instance" << std::endl;
            handle->models[model_name] = std::make_unique<inference::Model>(
                model_path, config.type, config, inference::DeviceType::GPU, 0);

            std::cerr << "Debug [InferenceLoadModel]: Loading model" << std::endl;
            if (!handle->models[model_name]->Load())
            {
                std::string err_msg = handle->models[model_name]->GetLastError();
                std::cerr << "Debug [InferenceLoadModel]: Load failed: " << err_msg << std::endl;
                handle->models.erase(model_name);
                if (error)
                    *error = strdup_helper(err_msg.c_str());
                return false;
            }

            std::cerr << "Debug [InferenceLoadModel]: Model loaded successfully" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Debug [InferenceLoadModel]: Exception: " << e.what() << std::endl;
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool InferenceUnloadModel(InferenceManagerHandle handle, const char *model_name, const char *version, ErrorMessage *error)
    {
        if (!handle || !model_name)
        {
            if (error)
                *error = strdup_helper("Invalid handle or model name");
            return false;
        }

        try
        {
            // Find the model
            auto it = handle->models.find(model_name);
            if (it == handle->models.end())
            {
                if (error)
                    *error = strdup_helper("Model not found");
                return false;
            }

            // Unload and remove the model
            handle->models.erase(it);
            return true;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool InferenceIsModelLoaded(InferenceManagerHandle handle, const char *model_name, const char *version)
    {
        if (!handle || !model_name)
        {
            return false;
        }

        try
        {
            // Find the model
            auto it = handle->models.find(model_name);
            return it != handle->models.end();
        }
        catch (...)
        {
            return false;
        }
    }

    char **InferenceListModels(InferenceManagerHandle handle, int *num_models)
    {
        if (!handle || !num_models)
        {
            return nullptr;
        }

        try
        {
            // Get models from repository instead of loaded models
            if (handle->repository)
            {
                // Make sure repository is scanned
                handle->repository->ScanRepository();

                // Get available models
                std::vector<std::string> available_models = handle->repository->GetAvailableModels();
                *num_models = static_cast<int>(available_models.size());

                if (*num_models == 0)
                {
                    return nullptr;
                }

                char **models = new char *[*num_models];
                for (int i = 0; i < *num_models; i++)
                {
                    models[i] = strdup_helper(available_models[i]);
                }
                return models;
            }
            else
            {
                // Fall back to listing loaded models if repository isn't available
                *num_models = static_cast<int>(handle->models.size());
                if (*num_models == 0)
                {
                    return nullptr;
                }

                char **models = new char *[*num_models];
                int i = 0;
                for (const auto &model : handle->models)
                {
                    models[i++] = strdup_helper(model.first);
                }
                return models;
            }
        }
        catch (...)
        {
            *num_models = 0;
            return nullptr;
        }
    }

    void InferenceFreeModelList(char **models, int num_models)
    {
        if (models)
        {
            for (int i = 0; i < num_models; i++)
            {
                free(models[i]);
            }
            delete[] models;
        }
    }

    // Model Functions
    ModelHandle ModelCreate(const char *model_path, ModelType type, const ModelConfig *config, DeviceType device, int device_id, ErrorMessage *error)
    {
        std::cerr << "DEBUG [ModelCreate]: Creating model for path " << (model_path ? model_path : "NULL")
                  << ", type " << static_cast<int>(type) << std::endl;

        if (!model_path || !config)
        {
            if (error)
                *error = strdup_helper("Invalid model path or configuration");
            return nullptr;
        }

        try
        {
            // Convert C model config to C++ model config
            inference::ModelConfig cpp_config;
            cpp_config.name = config->name ? config->name : "";
            cpp_config.version = config->version ? config->version : "1";
            cpp_config.type = convert_model_type(config->type_);
            cpp_config.max_batch_size = config->max_batch_size;
            cpp_config.instance_count = config->instance_count;
            cpp_config.dynamic_batching = config->dynamic_batching;

            // Copy input names
            for (int i = 0; i < config->num_inputs; i++)
            {
                if (config->input_names && config->input_names[i])
                {
                    cpp_config.input_names.push_back(config->input_names[i]);
                }
            }

            // Copy output names
            for (int i = 0; i < config->num_outputs; i++)
            {
                if (config->output_names && config->output_names[i])
                {
                    cpp_config.output_names.push_back(config->output_names[i]);
                }
            }

            // Create the model
            auto model = std::make_unique<inference::Model>(
                model_path,
                convert_model_type(type),
                cpp_config,
                convert_device_type(device),
                device_id);

            // Create and return bridge
            Model_t *handle = new Model_t(std::move(model));

            std::cerr << "DEBUG [ModelCreate]: Model created successfully: " << static_cast<void *>(handle) << std::endl;
            return handle;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return nullptr;
        }
    }

    void ModelDestroy(ModelHandle handle)
    {
        delete handle;
    }

    bool ModelLoad(ModelHandle handle, ErrorMessage *error)
    {
        std::cerr << "DEBUG [ModelLoad]: Loading model handle " << static_cast<void *>(handle) << std::endl;
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid model handle");
            std::cerr << "DEBUG [ModelLoad]: Invalid model handle" << std::endl;
            return false;
        }

        try
        {
            bool success = handle->model->Load();
            std::cerr << "DEBUG [ModelLoad]: handle->model->Load() returned " << (success ? "true" : "false") << std::endl;

            if (!success && error)
            {
                *error = strdup_helper(handle->model->GetLastError());
                std::cerr << "DEBUG [ModelLoad]: Error: " << handle->model->GetLastError() << std::endl;
            }

            // Check loaded state after loading
            std::cerr << "DEBUG [ModelLoad]: Model IsLoaded() = " << (handle->model->IsLoaded() ? "true" : "false") << std::endl;

            return success;
        }
        catch (const std::exception &e)
        {
            std::cerr << "DEBUG [ModelLoad]: Exception: " << e.what() << std::endl;
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool ModelUnload(ModelHandle handle, ErrorMessage *error)
    {
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid model handle");
            return false;
        }

        try
        {
            handle->model->Unload();
            return true;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool ModelIsLoaded(ModelHandle handle)
    {
        std::cerr << "DEBUG [ModelIsLoaded]: Checking if model is loaded for handle " << static_cast<void *>(handle) << std::endl;
        if (!handle)
        {
            std::cerr << "DEBUG [ModelIsLoaded]: Invalid handle, returning false" << std::endl;
            return false;
        }

        try
        {
            bool loaded = handle->model->IsLoaded();
            std::cerr << "DEBUG [ModelIsLoaded]: handle->model->IsLoaded() returned " << (loaded ? "true" : "false") << std::endl;
            return loaded;
        }
        catch (...)
        {
            std::cerr << "DEBUG [ModelIsLoaded]: Exception caught, returning false" << std::endl;
            return false;
        }
    }

    bool ModelInfer(ModelHandle handle, const TensorData *inputs, int num_inputs, TensorData *outputs, int num_outputs, ErrorMessage *error)
    {
        std::cerr << "DEBUG [ModelInfer]: Called with handle " << static_cast<void *>(handle)
                  << ", num_inputs=" << num_inputs << ", num_outputs=" << num_outputs << std::endl;

        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid model handle");
            std::cerr << "DEBUG [ModelInfer]: Invalid model handle" << std::endl;
            return false;
        }

        // Check if model is loaded
        bool isLoaded = handle->model->IsLoaded();
        std::cerr << "DEBUG [ModelInfer]: Model->IsLoaded() = " << (isLoaded ? "true" : "false") << std::endl;

        if (!isLoaded)
        {
            if (error)
                *error = strdup_helper("Model not loaded");
            std::cerr << "DEBUG [ModelInfer]: Returning error - Model not loaded" << std::endl;
            return false;
        }

        if (!handle || !inputs || num_inputs <= 0 || !outputs || num_outputs <= 0)
        {
            if (error)
                *error = strdup_helper("Invalid parameters");
            return false;
        }

        try
        {
            // Convert C input tensors to C++ tensors
            std::vector<inference::Tensor> cpp_inputs;
            cpp_inputs.reserve(num_inputs);

            for (int i = 0; i < num_inputs; i++)
            {
                const TensorData &input = inputs[i];

                // Convert shape
                inference::Shape shape;
                if (input.shape.dims && input.shape.num_dims > 0)
                {
                    shape.dims.resize(input.shape.num_dims);
                    for (int j = 0; j < input.shape.num_dims; j++)
                    {
                        shape.dims[j] = input.shape.dims[j];
                    }
                }

                // Create tensor
                inference::Tensor tensor(
                    input.name ? input.name : "",
                    convert_data_type(input.data_type),
                    shape);

                // Set data - this is simplified for float32 only
                if (input.data_type == DATATYPE_FLOAT32 && input.data && input.data_size > 0)
                {
                    size_t num_elements = shape.NumElements();
                    std::vector<float> data(num_elements);
                    memcpy(data.data(), input.data, std::min(input.data_size, num_elements * sizeof(float)));
                    tensor.SetData(data);
                }

                cpp_inputs.push_back(std::move(tensor));
            }

            // Prepare output tensors
            std::vector<inference::Tensor> cpp_outputs;
            cpp_outputs.reserve(num_outputs);

            for (int i = 0; i < num_outputs; i++)
            {
                const TensorData &output = outputs[i];

                // Convert shape
                inference::Shape shape;
                if (output.shape.dims && output.shape.num_dims > 0)
                {
                    shape.dims.resize(output.shape.num_dims);
                    for (int j = 0; j < output.shape.num_dims; j++)
                    {
                        shape.dims[j] = output.shape.dims[j];
                    }
                }

                // Create tensor
                inference::Tensor tensor(
                    output.name ? output.name : "",
                    convert_data_type(output.data_type),
                    shape);

                cpp_outputs.push_back(std::move(tensor));
            }

            // Run inference
            bool result = handle->model->Infer(cpp_inputs, cpp_outputs);

            // Copy output data back to C tensors
            for (int i = 0; i < num_outputs && i < cpp_outputs.size(); i++)
            {
                const inference::Tensor &cpp_output = cpp_outputs[i];
                TensorData &output = outputs[i];

                // Get shape
                const inference::Shape &cpp_shape = cpp_output.GetShape();
                output.shape.num_dims = cpp_shape.dims.size();

                // This assumes the dims array is pre-allocated by the caller
                if (output.shape.dims && output.shape.num_dims > 0)
                {
                    for (int j = 0; j < output.shape.num_dims; j++)
                    {
                        output.shape.dims[j] = cpp_shape.dims[j];
                    }
                }

                // Copy data - this is simplified for float32 only
                if (output.data_type == DATATYPE_FLOAT32 && output.data && output.data_size > 0)
                {
                    std::vector<float> data;
                    cpp_output.GetData(data);
                    size_t bytes_to_copy = std::min(output.data_size, data.size() * sizeof(float));
                    memcpy(output.data, data.data(), bytes_to_copy);
                }
            }

            if (!result && error)
            {
                *error = strdup_helper(handle->model->GetLastError());
            }

            return result;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    ModelMetadata *ModelGetMetadata(ModelHandle handle)
    {
        if (!handle)
        {
            return nullptr;
        }

        try
        {
            inference::ModelMetadata cpp_metadata = handle->model->GetMetadata();

            // Create C metadata struct
            ModelMetadata *metadata = new ModelMetadata();
            metadata->name = strdup_helper(cpp_metadata.name);
            metadata->version = strdup_helper(cpp_metadata.version);
            metadata->model_type = convert_to_c_model_type(cpp_metadata.type);
            metadata->description = strdup_helper(cpp_metadata.description);
            metadata->load_time_ns = cpp_metadata.load_time_ns;

            // Copy inputs
            metadata->num_inputs = cpp_metadata.inputs.size();
            if (metadata->num_inputs > 0)
            {
                metadata->inputs = new const char *[metadata->num_inputs];
                for (int i = 0; i < metadata->num_inputs; i++)
                {
                    metadata->inputs[i] = strdup_helper(cpp_metadata.inputs[i]);
                }
            }
            else
            {
                metadata->inputs = nullptr;
            }

            // Copy outputs
            metadata->num_outputs = cpp_metadata.outputs.size();
            if (metadata->num_outputs > 0)
            {
                metadata->outputs = new const char *[metadata->num_outputs];
                for (int i = 0; i < metadata->num_outputs; i++)
                {
                    metadata->outputs[i] = strdup_helper(cpp_metadata.outputs[i]);
                }
            }
            else
            {
                metadata->outputs = nullptr;
            }

            return metadata;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    void ModelFreeMetadata(ModelMetadata *metadata)
    {
        if (metadata)
        {
            free((void *)metadata->name);
            free((void *)metadata->version);
            free((void *)metadata->description);

            if (metadata->inputs)
            {
                for (int i = 0; i < metadata->num_inputs; i++)
                {
                    free((void *)metadata->inputs[i]);
                }
                delete[] metadata->inputs;
            }

            if (metadata->outputs)
            {
                for (int i = 0; i < metadata->num_outputs; i++)
                {
                    free((void *)metadata->outputs[i]);
                }
                delete[] metadata->outputs;
            }

            delete metadata;
        }
    }

    ModelStats *ModelGetStats(ModelHandle handle)
    {
        if (!handle)
        {
            return nullptr;
        }

        try
        {
            inference::Model::Stats cpp_stats = handle->model->GetStats();

            // Create C stats struct
            ModelStats *stats = new ModelStats();
            stats->inference_count = cpp_stats.inference_count;
            stats->total_inference_time_ns = cpp_stats.total_inference_time_ns;
            stats->last_inference_time_ns = cpp_stats.last_inference_time_ns;
            stats->memory_usage_bytes = cpp_stats.memory_usage_bytes;

            return stats;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    void ModelFreeStats(ModelStats *stats)
    {
        if (stats)
        {
            delete stats;
        }
    }

    // Tensor Functions - implemented without ResourceManager
    TensorHandle TensorCreate(const char *name, DataType data_type, const Shape *shape, ErrorMessage *error)
    {
        if (!name || !shape)
        {
            if (error)
                *error = strdup_helper("Invalid tensor parameters");
            return nullptr;
        }

        try
        {
            // Convert shape
            inference::Shape cpp_shape;
            if (shape->dims && shape->num_dims > 0)
            {
                cpp_shape.dims.resize(shape->num_dims);
                for (int i = 0; i < shape->num_dims; i++)
                {
                    cpp_shape.dims[i] = shape->dims[i];
                }
            }

            // Create tensor directly
            Tensor_t *handle = new Tensor_t(
                name,
                convert_data_type(data_type),
                cpp_shape);

            return handle;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return nullptr;
        }
    }

    void TensorDestroy(TensorHandle handle)
    {
        delete handle;
    }

    bool TensorSetData(TensorHandle handle, const void *data, size_t data_size, ErrorMessage *error)
    {
        if (!handle || !data || data_size <= 0)
        {
            if (error)
                *error = strdup_helper("Invalid tensor data");
            return false;
        }

        try
        {
            // This is simplified for float32 only
            if (handle->tensor.GetDataType() == inference::DataType::FLOAT32)
            {
                size_t num_elements = handle->tensor.GetShape().NumElements();
                std::vector<float> float_data(num_elements);
                memcpy(float_data.data(), data, std::min(data_size, num_elements * sizeof(float)));
                return handle->tensor.SetData(float_data);
            }
            else
            {
                if (error)
                    *error = strdup_helper("Unsupported data type");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool TensorGetData(TensorHandle handle, void *data, size_t data_size, ErrorMessage *error)
    {
        if (!handle || !data || data_size <= 0)
        {
            if (error)
                *error = strdup_helper("Invalid tensor data buffer");
            return false;
        }

        try
        {
            // This is simplified for float32 only
            if (handle->tensor.GetDataType() == inference::DataType::FLOAT32)
            {
                std::vector<float> float_data;
                bool result = handle->tensor.GetData(float_data);
                if (result)
                {
                    size_t bytes_to_copy = std::min(data_size, float_data.size() * sizeof(float));
                    memcpy(data, float_data.data(), bytes_to_copy);
                }
                return result;
            }
            else
            {
                if (error)
                    *error = strdup_helper("Unsupported data type");
                return false;
            }
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    Shape *TensorGetShape(TensorHandle handle)
    {
        if (!handle)
        {
            return nullptr;
        }

        try
        {
            const inference::Shape &cpp_shape = handle->tensor.GetShape();

            // Create C shape
            Shape *shape = new Shape();
            shape->num_dims = cpp_shape.dims.size();

            if (shape->num_dims > 0)
            {
                shape->dims = new int64_t[shape->num_dims];
                for (int i = 0; i < shape->num_dims; i++)
                {
                    shape->dims[i] = cpp_shape.dims[i];
                }
            }
            else
            {
                shape->dims = nullptr;
            }

            return shape;
        }
        catch (...)
        {
            return nullptr;
        }
    }

    void TensorFreeShape(Shape *shape)
    {
        if (shape)
        {
            delete[] shape->dims;
            delete shape;
        }
    }

    DataType TensorGetDataType(TensorHandle handle)
    {
        if (!handle)
        {
            return DATATYPE_UNKNOWN;
        }

        try
        {
            return convert_to_c_data_type(handle->tensor.GetDataType());
        }
        catch (...)
        {
            return DATATYPE_UNKNOWN;
        }
    }

    const char *TensorGetName(TensorHandle handle)
    {
        if (!handle)
        {
            return nullptr;
        }

        try
        {
            return strdup_helper(handle->tensor.GetName());
        }
        catch (...)
        {
            return nullptr;
        }
    }

    bool TensorToGPU(TensorHandle handle, int device_id, ErrorMessage *error)
    {
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid tensor handle");
            return false;
        }

        try
        {
            return handle->tensor.toGPU(device_id);
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    bool TensorToCPU(TensorHandle handle, ErrorMessage *error)
    {
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid tensor handle");
            return false;
        }

        try
        {
            return handle->tensor.toCPU();
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    // Utility Functions
    void FreeErrorMessage(ErrorMessage error)
    {
        free(error);
    }

    // Corrected GetModelHandle function in inference_bridge.cpp
    ModelHandle GetModelHandle(InferenceManagerHandle handle, const char *model_name, const char *version, ErrorMessage *error)
    {
        if (!handle || !model_name)
        {
            if (error)
                *error = strdup_helper("Invalid handle or model name");
            return nullptr;
        }

        try
        {
            // Find the model in the loaded models map
            auto it = handle->models.find(model_name);
            if (it == handle->models.end())
            {
                if (error)
                    *error = strdup_helper("Model not found in loaded models");
                return nullptr;
            }

            // Create a wrapper Model_t that doesn't own the model
            // but references the existing one
            Model_t *model_handle = new Model_t(nullptr);

            // Directly assign the model pointer to handle's model member
            // This bypasses unique_ptr ownership issues
            model_handle->model = it->second.get();

            return model_handle;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return nullptr;
        }
    }
}