// Bridge implementation between C and C++ for the inference server
#include "inference_bridge.h"
#include "model.h"
#include "cuda_utils.h"
#include "model_repository.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <cstring>

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
    inference::Model *model; // May point to owned_model.get() or an external model

    // Constructor for owned model
    Model_t(std::unique_ptr<inference::Model> m) : owned_model(std::move(m)), model(owned_model.get()) {}

    // Constructor for non-owned model reference
    Model_t(std::nullptr_t) : owned_model(nullptr), model(nullptr) {}

    // Destructor only deletes if we own the model
    ~Model_t()
    {
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

/**
 * Helper function to allocate C string copy
 *
 * @param str String to duplicate
 * @return Pointer to newly allocated C string that must be freed by caller
 */
char *strdup_helper(const std::string &str)
{
    char *result = strdup(str.c_str());
    return result;
}

/**
 * Helper function to convert from C ModelType to C++ ModelType
 *
 * @param type The C ModelType enum value
 * @return Corresponding C++ ModelType enum value
 */
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

/**
 * Helper function to convert from C DeviceType to C++ DeviceType
 *
 * @param device The C DeviceType enum value
 * @return Corresponding C++ DeviceType enum value
 */
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

/**
 * Helper function to convert from C DataType to C++ DataType
 *
 * @param dtype The C DataType enum value
 * @return Corresponding C++ DataType enum value
 */
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

/**
 * Helper function to convert from C++ ModelType to C ModelType
 *
 * @param type The C++ ModelType enum value
 * @return Corresponding C ModelType enum value
 */
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

/**
 * Helper function to convert from C++ DataType to C DataType
 *
 * @param dtype The C++ DataType enum value
 * @return Corresponding C DataType enum value
 */
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
    /**
     * Check if CUDA is available on the system
     *
     * @return true if CUDA is available, false otherwise
     */
    bool IsCudaAvailable()
    {
        return inference::cuda::IsCudaAvailable();
    }

    /**
     * Get the number of CUDA devices (GPUs) available on the system
     *
     * @return Number of CUDA devices, 0 if no devices are available
     */
    int GetDeviceCount()
    {
        return inference::cuda::GetDeviceCount();
    }

    /**
     * Get information about a specific CUDA device
     *
     * @param device_id The ID of the device to query
     * @return String containing device information (must be freed by caller)
     */
    const char *GetDeviceInfo(int device_id)
    {
        std::string info = inference::cuda::GetDeviceInfo(device_id);
        return strdup_helper(info);
    }

    /**
     * Get memory information for a specific CUDA device
     *
     * @param device_id The ID of the device to query
     * @return CudaMemoryInfo structure containing total, free, and used memory
     */
    CudaMemoryInfo GetMemoryInfo(int device_id)
    {
        inference::cuda::MemoryInfo memInfo = inference::cuda::GetMemoryInfo(device_id);

        CudaMemoryInfo cInfo;
        cInfo.total = memInfo.total;
        cInfo.free = memInfo.free;
        cInfo.used = memInfo.used;

        return cInfo;
    }

    /**
     * Initialize the inference manager with a model repository
     *
     * @param model_repository_path Path to the model repository
     * @return Handle to the inference manager, NULL if initialization fails
     */
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

    /**
     * Shutdown and clean up the inference manager
     *
     * @param handle Handle to the inference manager
     */
    void InferenceShutdown(InferenceManagerHandle handle)
    {
        delete handle;
    }

    /**
     * Load a model from the repository
     *
     * @param handle Handle to the inference manager
     * @param model_name Name of the model to load
     * @param version Model version to load, NULL or empty string for latest version
     * @param error If error occurs, will contain error message that must be freed
     * @return true if model loaded successfully, false otherwise
     */
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
            // Get model path and configuration from repository
            std::string resolved_version = version ? version : handle->repository->GetLatestVersion(model_name);
            std::string model_path = handle->repository->GetModelPath(model_name, resolved_version);

            // Verify that the model directory exists
            if (!std::filesystem::exists(model_path))
            {
                if (error)
                    *error = strdup_helper(("Model path not found: " + model_path).c_str());
                return false;
            }

            // Check if model is already loaded
            if (handle->models.find(model_name) != handle->models.end())
            {
                if (error)
                    *error = strdup_helper("Model already loaded");
                return false;
            }

            // Check for model.onnx file
            std::string onnx_file_path = model_path + "/model.onnx";
            if (!std::filesystem::exists(onnx_file_path))
            {
                if (error)
                    *error = strdup_helper(("ONNX file not found at: " + onnx_file_path).c_str());
                return false;
            }

            // Load the model configuration from repository
            inference::ModelConfig config = handle->repository->GetModelConfig(model_name, resolved_version);
            if (config.type == inference::ModelType::UNKNOWN)
            {
                if (error)
                    *error = strdup_helper("Unable to determine model type");
                return false;
            }

            // Create and load the model
            handle->models[model_name] = std::make_unique<inference::Model>(
                model_path, config.type, config, inference::DeviceType::GPU, 0);

            if (!handle->models[model_name]->Load())
            {
                std::string err_msg = handle->models[model_name]->GetLastError();
                handle->models.erase(model_name);
                if (error)
                    *error = strdup_helper(err_msg.c_str());
                return false;
            }

            return true;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    /**
     * Unload a model
     *
     * @param handle Handle to the inference manager
     * @param model_name Name of the model to unload
     * @param version Model version to unload, NULL or empty string for latest version
     * @param error If error occurs, will contain error message that must be freed
     * @return true if model unloaded successfully, false otherwise
     */
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

    /**
     * Check if a model is loaded
     *
     * @param handle Handle to the inference manager
     * @param model_name Name of the model to check
     * @param version Model version to check, NULL or empty string for latest version
     * @return true if model is loaded, false otherwise
     */
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

    /**
     * List available models in the repository
     *
     * @param handle Handle to the inference manager
     * @param num_models Output parameter that will be set to the number of models
     * @return Array of model name strings that must be freed with InferenceFreeModelList
     */
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

    /**
     * Free a list of models returned by InferenceListModels
     *
     * @param models Array of model name strings
     * @param num_models Number of models in the array
     */
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

    /**
     * Create a model instance
     *
     * @param model_path Path to the model file
     * @param type Type of the model (TensorFlow, PyTorch, etc.)
     * @param config Configuration for the model
     * @param device Device to run model on (CPU or GPU)
     * @param device_id ID of the device to use
     * @param error If error occurs, will contain error message that must be freed
     * @return Handle to the model, NULL if creation fails
     */
    ModelHandle ModelCreate(const char *model_path, ModelType type, const ModelConfig *config, DeviceType device, int device_id, ErrorMessage *error)
    {
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
            return handle;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return nullptr;
        }
    }

    /**
     * Destroy a model instance
     *
     * @param handle Handle to the model
     */
    void ModelDestroy(ModelHandle handle)
    {
        delete handle;
    }

    /**
     * Load a model into memory
     *
     * @param handle Handle to the model
     * @param error If error occurs, will contain error message that must be freed
     * @return true if model loaded successfully, false otherwise
     */
    bool ModelLoad(ModelHandle handle, ErrorMessage *error)
    {
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid model handle");
            return false;
        }

        try
        {
            bool success = handle->model->Load();
            if (!success && error)
            {
                *error = strdup_helper(handle->model->GetLastError());
            }
            return success;
        }
        catch (const std::exception &e)
        {
            if (error)
                *error = strdup_helper(e.what());
            return false;
        }
    }

    /**
     * Unload a model from memory
     *
     * @param handle Handle to the model
     * @param error If error occurs, will contain error message that must be freed
     * @return true if model unloaded successfully, false otherwise
     */
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

    /**
     * Check if a model is loaded
     *
     * @param handle Handle to the model
     * @return true if model is loaded, false otherwise
     */
    bool ModelIsLoaded(ModelHandle handle)
    {
        if (!handle)
        {
            return false;
        }

        try
        {
            return handle->model->IsLoaded();
        }
        catch (...)
        {
            return false;
        }
    }

    /**
     * Run inference on a model
     *
     * @param handle Handle to the model
     * @param inputs Array of input tensors
     * @param num_inputs Number of input tensors
     * @param outputs Array of output tensors to be filled
     * @param num_outputs Number of output tensors
     * @param error If error occurs, will contain error message that must be freed
     * @return true if inference was successful, false otherwise
     */
    bool ModelInfer(ModelHandle handle, const TensorData *inputs, int num_inputs, TensorData *outputs, int num_outputs, ErrorMessage *error)
    {
        if (!handle)
        {
            if (error)
                *error = strdup_helper("Invalid model handle");
            return false;
        }

        // Check if model is loaded
        if (!handle->model->IsLoaded())
        {
            if (error)
                *error = strdup_helper("Model not loaded");
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

    /**
     * Get metadata about a model
     *
     * @param handle Handle to the model
     * @return Metadata structure, NULL if not available. Must be freed with ModelFreeMetadata
     */
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

    /**
     * Free a model metadata structure
     *
     * @param metadata Metadata structure to free
     */
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

    /**
     * Get statistics about a model
     *
     * @param handle Handle to the model
     * @return Statistics structure, NULL if not available. Must be freed with ModelFreeStats
     */
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

    /**
     * Free a model statistics structure
     *
     * @param stats Statistics structure to free
     */
    void ModelFreeStats(ModelStats *stats)
    {
        if (stats)
        {
            delete stats;
        }
    }

    /**
     * Free an error message
     *
     * @param error Error message to free
     */
    void FreeErrorMessage(ErrorMessage error)
    {
        free(error);
    }

    /**
     * Get a handle to an already loaded model
     *
     * @param handle Handle to the inference manager
     * @param model_name Name of the model to get
     * @param version Model version to get, NULL or empty string for latest version
     * @param error If error occurs, will contain error message that must be freed
     * @return Handle to the model, NULL if not found
     */
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