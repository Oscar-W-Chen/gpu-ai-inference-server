#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include "model.h"

namespace inference {

// Model state to track lifecycle
enum class ModelState {
    UNLOADED,     // Model is not loaded
    LOADING,      // Model is in the process of loading
    LOADED,       // Model is loaded and ready for inference
    UNLOADING,    // Model is in the process of unloading
    ERROR         // Model is in an error state
};

// Forward declaration
class ModelRepository;

/**
 * @class InferenceManager
 * @brief Manages model loading, inference and tracking
 * 
 * The InferenceManager is responsible for managing the lifecycle of models, 
 * including loading, unloading, and executing inference requests. It serves
 * as the central coordination point for the inference server.
 */
class InferenceManager {
public:
    /**
     * @brief Constructor
     * @param model_repository_path Path to the model repository
     */
    InferenceManager(const std::string& model_repository_path);
    
    /**
     * @brief Destructor
     */
    ~InferenceManager();
    
    /**
     * @brief Initialize the inference manager
     * @return true if initialization was successful, false otherwise
     */
    bool Initialize();
    
    /**
     * @brief Shutdown the inference manager
     */
    void Shutdown();
    
    /**
     * @brief Load a model
     * @param model_name Name of the model to load
     * @param version Version of the model to load (empty for latest)
     * @return true if loading was successful, false otherwise
     */
    bool LoadModel(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Unload a model
     * @param model_name Name of the model to unload
     * @param version Version of the model to unload (empty for latest)
     * @return true if unloading was successful, false otherwise
     */
    bool UnloadModel(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Check if a model is loaded
     * @param model_name Name of the model to check
     * @param version Version of the model to check (empty for latest)
     * @return true if the model is loaded, false otherwise
     */
    bool IsModelLoaded(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Get the current state of a model
     * @param model_name Name of the model
     * @param version Version of the model (empty for latest)
     * @return The current state of the model
     */
    ModelState GetModelState(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief List all available models
     * @return Vector of model names
     */
    std::vector<std::string> ListModels();
    
    /**
     * @brief Get a model by name
     * @param model_name Name of the model to get
     * @param version Version of the model to get (empty for latest)
     * @return Pointer to the model, or nullptr if not found
     */
    std::shared_ptr<Model> GetModel(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Run inference with a model
     * @param model_name Name of the model to use
     * @param version Version of the model to use (empty for latest)
     * @param inputs Vector of input tensors
     * @param outputs Vector of output tensors to be filled
     * @return true if inference was successful, false otherwise
     */
    bool RunInference(const std::string& model_name, 
                      const std::string& version,
                      const std::vector<Tensor>& inputs, 
                      std::vector<Tensor>& outputs);
    
    /**
     * @brief Get the last error message
     * @return Last error message
     */
    std::string GetLastError() const;

private:
    // Path to model repository
    std::string model_repository_path_;
    
    // Model repository manager
    std::unique_ptr<ModelRepository> repository_;
    
    // Model info struct to track state
    struct ModelInfo {
        std::shared_ptr<Model> model;
        ModelState state;
        std::string error_message;
        std::chrono::time_point<std::chrono::system_clock> state_changed_time;
    };
    
    // Map of models - key is "name:version"
    std::unordered_map<std::string, ModelInfo> models_;
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
    
    // Last error message
    mutable std::string last_error_;
    
    // Internal helper to create a model key
    std::string MakeModelKey(const std::string& name, const std::string& version) const;
    
    // Set error message
    void SetError(const std::string& error) const;
};

} // namespace inference

#endif // INFERENCE_MANAGER_H