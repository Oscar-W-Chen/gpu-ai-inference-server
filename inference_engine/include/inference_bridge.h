#ifndef INFERENCE_BRIDGE_H
#define INFERENCE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// Opaque handle types
typedef struct InferenceManager_t* InferenceManagerHandle;
typedef struct Model_t* ModelHandle;
typedef struct Tensor_t* TensorHandle;

// Error handling
typedef char* ErrorMessage;  // Caller must free with FreeErrorMessage

// Data type enum (must match DataType in model.h)
typedef enum {
    DATATYPE_FLOAT32 = 0,
    DATATYPE_INT32,
    DATATYPE_INT64,
    DATATYPE_UINT8,
    DATATYPE_INT8,
    DATATYPE_STRING,
    DATATYPE_BOOL,
    DATATYPE_FP16,
    DATATYPE_UNKNOWN
} DataType;

// Device type enum (must match DeviceType in model.h)
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_GPU
} DeviceType;

// Model type enum (must match ModelType in model.h)
typedef enum {
    MODEL_UNKNOWN = 0,
    MODEL_TENSORFLOW,
    MODEL_TENSORRT,
    MODEL_ONNX,
    MODEL_PYTORCH,
    MODEL_CUSTOM
} ModelType;

// Shape struct
typedef struct {
    int64_t* dims;
    int num_dims;
} Shape;

// Tensor struct
typedef struct {
    const char* name;
    DataType data_type;
    Shape shape;
    void* data;
    size_t data_size;
} TensorData;

// Model config struct
typedef struct {
    const char* name;
    const char* version;
    ModelType type_;
    int max_batch_size;
    const char** input_names;
    int num_inputs;
    const char** output_names;
    int num_outputs;
    int instance_count;
    bool dynamic_batching;
} ModelConfig;

// Model metadata struct
typedef struct {
    const char* name;
    const char* version;
    ModelType model_type;
    const char** inputs;
    int num_inputs;
    const char** outputs;
    int num_outputs;
    const char* description;
    int64_t load_time_ns;
} ModelMetadata;

// Model statistics struct
typedef struct {
    int64_t inference_count;
    int64_t total_inference_time_ns;
    int64_t last_inference_time_ns;
    size_t memory_usage_bytes;
} ModelStats;

// Memory information structure
typedef struct {
    size_t total;
    size_t free;
    size_t used;
} CudaMemoryInfo;

// CUDA utility functions
bool IsCudaAvailable();
int GetDeviceCount();
const char* GetDeviceInfo(int device_id);
CudaMemoryInfo GetMemoryInfo(int device_id);

// Inference Manager functions
InferenceManagerHandle InferenceInitialize(const char* model_repository_path);
void InferenceShutdown(InferenceManagerHandle handle);
bool InferenceLoadModel(InferenceManagerHandle handle, const char* model_name, const char* version, ErrorMessage* error);
bool InferenceUnloadModel(InferenceManagerHandle handle, const char* model_name, const char* version, ErrorMessage* error);
bool InferenceIsModelLoaded(InferenceManagerHandle handle, const char* model_name, const char* version);
char** InferenceListModels(InferenceManagerHandle handle, int* num_models);
void InferenceFreeModelList(char** models, int num_models);

// Model functions
ModelHandle ModelCreate(const char* model_path, ModelType type, const ModelConfig* config, DeviceType device, int device_id, ErrorMessage* error);
void ModelDestroy(ModelHandle handle);
bool ModelIsLoaded(ModelHandle handle);
bool ModelInfer(ModelHandle handle, const TensorData* inputs, int num_inputs, TensorData* outputs, int num_outputs, ErrorMessage* error);
ModelMetadata* ModelGetMetadata(ModelHandle handle);
void ModelFreeMetadata(ModelMetadata* metadata);
ModelStats* ModelGetStats(ModelHandle handle);
void ModelFreeStats(ModelStats* stats);

// Utility functions
void FreeErrorMessage(ErrorMessage error);
ModelHandle GetModelHandle(InferenceManagerHandle handle, const char* model_name, const char* version, ErrorMessage* error);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_BRIDGE_H