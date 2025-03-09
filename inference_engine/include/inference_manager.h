#ifndef INFERENCE_MANAGER_H
#define INFERENCE_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <mutex>
#include <future>
#include <functional>
#include <thread>
#include <condition_variable>
#include <queue>
#include <atomic>
#include "model.h"

namespace inference {

/**
 * @brief Model state to track lifecycle
 */
enum class ModelState {
    UNAVAILABLE,  // Model not available (not in repository or error reading)
    UNLOADED,     // Model is in repository but not loaded
    LOADING,      // Model is in the process of loading asynchronously
    LOADED,       // Model is loaded and ready for inference
    UNLOADING,    // Model is in the process of unloading asynchronously
    ERROR         // Model is in an error state
};

/**
 * @brief Convert model state to string representation
 * @param state ModelState enum value
 * @return String representation
 */
std::string ModelStateToString(ModelState state);

// Forward declaration
class ModelRepository;

/**
 * @brief Callback function type for async operations
 * @param success True if operation succeeded
 * @param model_key Model identifier string
 * @param error_msg Error message if operation failed
 */
using ModelOperationCallback = std::function<void(bool success, const std::string& model_key, const std::string& error_msg)>;

/**
 * @class InferenceManager
 * @brief Manages model loading, inference and tracking with async capabilities
 * 
 * The InferenceManager is responsible for managing the lifecycle of models, 
 * including asynchronous loading/unloading, and executing inference requests.
 * It serves as the central coordination point for the inference server.
 */
class InferenceManager {
public:
    /**
     * @brief Constructor
     * @param model_repository_path Path to the model repository
     * @param num_worker_threads Number of worker threads for async operations
     */
    InferenceManager(const std::string& model_repository_path, 
                    int num_worker_threads = 4);
    
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
     * @brief Load a model synchronously
     * @param model_name Name of the model to load
     * @param version Version of the model to load (empty for latest)
     * @return true if loading was successful/queued, false if error
     */
    bool LoadModel(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Load a model asynchronously
     * @param model_name Name of the model to load
     * @param version Version of the model to load (empty for latest)
     * @param callback Function to call when loading completes
     * @return true if loading was queued successfully, false otherwise
     */
    bool LoadModelAsync(const std::string& model_name, 
                       const std::string& version = "",
                       ModelOperationCallback callback = nullptr);
    
    /**
     * @brief Unload a model synchronously
     * @param model_name Name of the model to unload
     * @param version Version of the model to unload (empty for latest)
     * @return true if unloading was successful/queued, false if error
     */
    bool UnloadModel(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief Unload a model asynchronously
     * @param model_name Name of the model to unload
     * @param version Version of the model to unload (empty for latest)
     * @param callback Function to call when unloading completes
     * @return true if unloading was queued successfully, false otherwise
     */
    bool UnloadModelAsync(const std::string& model_name, 
                         const std::string& version = "",
                         ModelOperationCallback callback = nullptr);
    
    /**
     * @brief Check if a model is loaded and ready for inference
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
     * @brief Get detailed model status including error messages and timestamps
     * @param model_name Name of the model
     * @param version Version of the model (empty for latest)
     * @return JSON string with model status details
     */
    std::string GetModelStatus(const std::string& model_name, const std::string& version = "");
    
    /**
     * @brief List all available models in the repository
     * @return Vector of model names
     */
    std::vector<std::string> ListModels();
    
    /**
     * @brief Get a model by name if it's loaded
     * @param model_name Name of the model to get
     * @param version Version of the model to get (empty for latest)
     * @return Pointer to the model, or nullptr if not found or not loaded
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
    
    // Model info struct to track state and async operations
    struct ModelInfo {
        std::shared_ptr<Model> model;
        ModelState state;
        std::string error_message;
        std::chrono::time_point<std::chrono::system_clock> state_changed_time;
        
        // For async operations
        std::future<bool> operation_future;
        ModelOperationCallback completion_callback;
        
        // Default constructor
        ModelInfo() : state(ModelState::UNLOADED) {
            state_changed_time = std::chrono::system_clock::now();
        }
    };
    
    // Map of models - key is "name:version"
    std::unordered_map<std::string, ModelInfo> models_;
    
    // Thread pool for async operations
    struct AsyncTask {
        enum class TaskType { LOAD, UNLOAD };
        TaskType type;
        std::string model_key;
        std::string model_name;
        std::string version;
        ModelOperationCallback callback;
    };
    
    std::vector<std::thread> worker_threads_;
    std::queue<AsyncTask> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::atomic<bool> shutdown_flag_;
    
    // Mutex for thread safety of model operations
    mutable std::mutex models_mutex_;
    
    // Last error message
    mutable std::string last_error_;
    mutable std::mutex error_mutex_;
    
    // Internal helper methods
    
    // Create a model key from name and version
    std::string MakeModelKey(const std::string& name, const std::string& version) const;
    
    // Set error message (thread-safe)
    void SetError(const std::string& error) const;
    
    // Worker thread function for async operations
    void WorkerThreadFunc();
    
    // Internal implementation of model loading
    bool LoadModelInternal(const std::string& model_name, 
                          const std::string& version,
                          const std::string& model_key);
    
    // Internal implementation of model unloading
    bool UnloadModelInternal(const std::string& model_name,
                            const std::string& version,
                            const std::string& model_key);
};

} // namespace inference

#endif // INFERENCE_MANAGER_H