#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace inference {

// For declaration for implementation
class ModelImpl;

// Model types supported by the inference server
enum class ModelType {
    UNKNOWN,
    TENSORFLOW,
    TENSORRT,
    ONNX,
    PYTORCH,
    CUSTOM
};

// Device type for model execution
enum class DeviceType {
    CPU,
    GPU
};

// Shape information for a tensor
struct Shape {
    std::vector<int64_t> dims;

    // Helper to get total element count
    size_t NumElements() const {
        if (dims.empty()) return 0;
        size_t count = 1;
        for (auto dim: dims) {
            count *= dim;
        }
        return count;
    }
};

// Data types supported for tensors
enum class DataType {
    FLOAT32,
    INT32,
    INT64,
    UINT8,
    INT8,
    STRING,
    BOOL,
    FP16,
    UNKNOWN
};

// Model configuration
struct ModelConfig {
    std::string name;
    std::string version;
    ModelType type;
    int max_batch_size;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::unordered_map<std::string, Shape> input_shapes;
    std::unordered_map<std::string, Shape> output_shapes;
    std::unordered_map<std::string, DataType> input_types;
    std::unordered_map<std::string, DataType> output_types;
    int instance_count;
    bool dynamic_batching;

    // Default constructor
    ModelConfig() :
        type(ModelType::UNKNOWN),
        max_batch_size(0),
        instance_count(1),
        dynamic_batching(false) {}
};

// Struct for metadata about a loaded model
struct ModelMetadata {
    std::string name;
    std::string version;
    ModelType type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string description;
    int64_t load_time_ns;
};

// Tensor structure for model inputs/outputs
class Tensor {
public:
    Tensor();
    Tensor(const std::string& name, DataType dtype, const Shape& shape);
    ~Tensor();

    // Set data for the tensor (copies input data)
    template<typename T>
    bool SetData(const std::vector<T>& data);

    // Get data from the tensor (copies output data)
    template<typename T>
    bool GetData(std::vector<T>& data) const;

    // Get tensor properties
    const std::string& GetName() const;
    DataType GetDataType() const;
    const Shape& GetShape() const;

    // Change tensor shape
    bool Reshape(const Shape& new_shape);

    // Move tensor to CPU/GPU
    bool toGPU(int device_id = 0);
    bool toCPU();

private:
    class TensorImpl;
    std::unique_ptr<TensorImpl> impl_;
};

// Main model class
class Model {
public:
    // Constructor with model path
    Model(const std::string& model_path,
          ModelType type,
          const ModelConfig& config,
          DeviceType device = DeviceType::GPU,
          int device_id = 0);
    
    // Destructor
    ~Model();

    // Disable Copy
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Move constructor and assignment
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;

    //Load model into memory
    bool Load();

    // Run inference on given inputs
    bool Infer(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

    // Get model metadata
    ModelMetadata GetMetadata() const;

    // Check if model is loaded
    bool IsLoaded() const;

    // Unload model from memory
    void Unload();

    // Get last error message
    std::string GetLastError() const;

    // Get model statistics
    struct Stats {
        int64_t inference_count;
        int64_t total_inference_time_ns;
        int64_t last_inference_time_ns;
        size_t memory_usage_bytes;
    };

    Stats GetStats() const;

private:
    // Private implementation (PIMPL (Pointer to Implementation) pattern)
    std::unique_ptr<ModelImpl> impl_;
};

} // namespace inference

#endif // MODEL_H