#include "inference_manager.h"
#include "cuda_utils.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>

namespace inference {

//==============================================================================
// String Conversion Functions
//==============================================================================

/**
 * @brief Convert model state enum to string representation
 */
std::string ModelStateToString(ModelState state) {
    switch (state) {
        case ModelState::UNAVAILABLE: return "UNAVAILABLE";
        case ModelState::UNLOADED: return "UNLOADED";
        case ModelState::LOADING: return "LOADING";
        case ModelState::LOADED: return "LOADED";
        case ModelState::UNLOADING: return "UNLOADING";
        case ModelState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

//==============================================================================
// ModelRepository Class Implementation
//==============================================================================
/**
 * @brief Manages access to models stored in the filesystem
 */
class ModelRepository {
public:
    /**
     * @brief Constructor with repository path
     */
    ModelRepository(const std::string& path) : repository_path_(path) {}

    /**
     * @brief Scan the repository for models and versions
     */
    bool ScanRepository() {
        try {
            // Verify repository path exists
            if (!std::filesystem::exists(repository_path_)) {
                std::cerr << "Model repository path does not exist: " << repository_path_ << std::endl;
                return false;
            }

            // Clear existing model info before scanning
            model_versions_.clear();

            // First level directories are model names
            for (const auto& model_dir : std::filesystem::directory_iterator(repository_path_)) {
                if (model_dir.is_directory()) {
                    std::string model_name = model_dir.path().filename().string();
                    std::vector<std::string> versions;

                    // Second level directories are version numbers
                    for (const auto& version_dir : std::filesystem::directory_iterator(model_dir.path())) {
                        if (version_dir.is_directory()) {
                            std::string version = version_dir.path().filename().string();
                            
                            // Check if this directory contains model files or configuration
                            if (HasModelConfig(version_dir.path())) {
                                versions.push_back(version);
                            }
                        }
                    }

                    // Sort versions in descending order (newest first)
                    std::sort(versions.begin(), versions.end(), 
                        [](const std::string& a, const std::string& b) {
                            try {
                                return std::stoi(a) > std::stoi(b);
                            } catch (const std::exception&) {
                                // Handle non-numeric versions
                                return a > b;
                            }
                        });

                    // Only add models that have at least one valid version
                    if (!versions.empty()) {
                        model_versions_[model_name] = versions;
                    }
                }
            }

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error scanning repository: " << e.what() << std::endl;
            return false;
        }
    }

    /**
     * @brief Check if a model exists in the repository
     */
    bool ModelExists(const std::string& model_name, const std::string& version = "") const {
        auto it = model_versions_.find(model_name);
        if (it == model_versions_.end() || it->second.empty()) {
            return false;
        }

        if (version.empty()) {
            // Any version is fine
            return true;
        }

        // Check specific version
        return std::find(it->second.begin(), it->second.end(), version) != it->second.end();
    }

    /**
     * @brief Get list of all available model names
     */
    std::vector<std::string> GetAvailableModels() const {
        std::vector<std::string> models;
        for (const auto& model : model_versions_) {
            models.push_back(model.first);
        }
        return models;
    }

    /**
     * @brief Get the filesystem path to a model
     */
    std::string GetModelPath(const std::string& model_name, const std::string& version = "") const {
        // Check if model exists in our registry
        auto it = model_versions_.find(model_name);
        if (it == model_versions_.end() || it->second.empty()) {
            return "";
        }

        // Determine which version to use
        std::string model_version = version;
        if (model_version.empty()) {
            // Use latest version if none specified
            model_version = it->second.front();
        } else {
            // Check if requested version exists
            auto version_it = std::find(it->second.begin(), it->second.end(), model_version);
            if (version_it == it->second.end()) {
                return "";
            }
        }

        // Construct and return the full path
        return std::filesystem::path(repository_path_) / model_name / model_version;
    }

    /**
     * @brief Get model configuration
     */
    ModelConfig GetModelConfig(const std::string& model_name, const std::string& version = "") const {
        ModelConfig config;
        std::string model_path = GetModelPath(model_name, version);
        
        if (model_path.empty()) {
            return config;
        }

        // Set basic model properties
        config.name = model_name;
        config.version = version.empty() ? GetLatestVersion(model_name) : version;
        
        // Try to infer model type based on available files
        auto model_type = DetectModelType(model_path);
        config.type = model_type;

        // Add default inputs and outputs
        // In a real implementation, these would be parsed from a config file
        config.input_names = {"input"};
        config.output_names = {"output"};

        return config;
    }

    /**
     * @brief Get the latest version for a model
     */
    std::string GetLatestVersion(const std::string& model_name) const {
        auto it = model_versions_.find(model_name);
        if (it == model_versions_.end() || it->second.empty()) {
            return "";
        }
        // First version in the list is the latest (due to sorting in ScanRepository)
        return it->second.front();
    }

    /**
     * @brief Get all available versions for a model
     */
    std::vector<std::string> GetModelVersions(const std::string& model_name) const {
        auto it = model_versions_.find(model_name);
        if (it == model_versions_.end()) {
            return {};
        }
        return it->second;
    }

private:
    // Path to model repository directory
    std::string repository_path_;
    
    // Map of model names to their available versions (sorted by version)
    std::unordered_map<std::string, std::vector<std::string>> model_versions_;

    /**
     * @brief Check if a directory contains valid model files
     */
    bool HasModelConfig(const std::filesystem::path& model_path) const {
        // Check for various types of model files or configuration
        return std::filesystem::exists(model_path / "config.pbtxt") ||
               std::filesystem::exists(model_path / "model.onnx") ||
               std::filesystem::exists(model_path / "model.pt") ||
               std::filesystem::exists(model_path / "saved_model.pb");
    }

    /**
     * @brief Attempt to detect model type from files in directory
     */
    ModelType DetectModelType(const std::filesystem::path& model_path) const {
        // Identify model type based on file extensions
        if (std::filesystem::exists(model_path / "saved_model.pb")) {
            return ModelType::TENSORFLOW;
        } else if (std::filesystem::exists(model_path / "model.plan")) {
            return ModelType::TENSORRT;
        } else if (std::filesystem::exists(model_path / "model.onnx")) {
            return ModelType::ONNX;
        } else if (std::filesystem::exists(model_path / "model.pt")) {
            return ModelType::PYTORCH;
        }
        return ModelType::UNKNOWN;
    }
};

//==============================================================================
// InferenceManager Implementation
//==============================================================================

/**
 * @brief Constructor
 */
InferenceManager::InferenceManager(const std::string& model_repository_path, int num_worker_threads)
    : model_repository_path_(model_repository_path),
      shutdown_flag_(false) {
    
    // Create worker threads for async operations
    for (int i = 0; i < num_worker_threads; i++) {
        worker_threads_.emplace_back(&InferenceManager::WorkerThreadFunc, this);
    }
}

/**
 * @brief Destructor
 */
InferenceManager::~InferenceManager() {
    Shutdown();
}

/**
 * @brief Initialize the inference manager and repository
 */
bool InferenceManager::Initialize() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    try {
        // Create model repository manager
        repository_ = std::make_unique<ModelRepository>(model_repository_path_);
        
        // Scan for available models
        if (!repository_->ScanRepository()) {
            SetError("Failed to scan model repository");
            return false;
        }
        
        return true;
    } catch (const std::exception& e) {
        SetError(std::string("Initialization error: ") + e.what());
        return false;
    }
}

/**
 * @brief Shutdown and release all resources
 */
void InferenceManager::Shutdown() {
    // Signal worker threads to stop
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        shutdown_flag_ = true;
    }
    
    // Wake up all worker threads
    queue_condition_.notify_all();
    
    // Wait for all worker threads to finish
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Unload all models
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        models_.clear();
        repository_.reset();
    }
}

/**
 * @brief Worker thread function for async operations
 */
void InferenceManager::WorkerThreadFunc() {
    while (true) {
        AsyncTask task;
        
        // Wait for a task
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait(lock, [this] {
                return shutdown_flag_ || !task_queue_.empty();
            });
            
            // Check if we should exit
            if (shutdown_flag_ && task_queue_.empty()) {
                break;
            }
            
            // Get the next task
            if (!task_queue_.empty()) {
                task = task_queue_.front();
                task_queue_.pop();
            } else {
                continue;
            }
        }
        
        // Process the task
        bool success = false;
        std::string error_msg;
        
        try {
            if (task.type == AsyncTask::TaskType::LOAD) {
                success = LoadModelInternal(task.model_name, task.version, task.model_key);
                if (!success) {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    error_msg = last_error_;
                }
            } else if (task.type == AsyncTask::TaskType::UNLOAD) {
                success = UnloadModelInternal(task.model_name, task.version, task.model_key);
                if (!success) {
                    std::lock_guard<std::mutex> lock(error_mutex_);
                    error_msg = last_error_;
                }
            }
        } catch (const std::exception& e) {
            success = false;
            error_msg = e.what();
            SetError(error_msg);
        }
        
        // Call the callback if provided
        if (task.callback) {
            try {
                task.callback(success, task.model_key, error_msg);
            } catch (const std::exception& e) {
                std::cerr << "Exception in model operation callback: " << e.what() << std::endl;
            }
        }
    }
}

/**
 * @brief Create a unique key for a model+version combination
 */
std::string InferenceManager::MakeModelKey(const std::string& name, const std::string& version) const {
    if (version.empty()) {
        // If version is empty, use the latest version if available
        if (repository_) {
            std::string latest = repository_->GetLatestVersion(name);
            if (!latest.empty()) {
                return name + ":" + latest;
            }
        }
        return name;
    }
    return name + ":" + version;
}

/**
 * @brief Load a model synchronously
 */
bool InferenceManager::LoadModel(const std::string& model_name, const std::string& version) {
    std::string model_key = MakeModelKey(model_name, version);
    
    // Check if model exists in repository
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        if (!repository_->ModelExists(model_name, version.empty() ? repository_->GetLatestVersion(model_name) : version)) {
            SetError("Model not found in repository: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
    }
    
    return LoadModelInternal(model_name, version, model_key);
}

/**
 * @brief Load a model asynchronously
 */
bool InferenceManager::LoadModelAsync(const std::string& model_name, 
                                     const std::string& version,
                                     ModelOperationCallback callback) {
    std::string model_key = MakeModelKey(model_name, version);
    
    // Check if model exists in repository
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        if (!repository_->ModelExists(model_name, version.empty() ? repository_->GetLatestVersion(model_name) : version)) {
            SetError("Model not found in repository: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
    }
    
    // Create async task
    AsyncTask task;
    task.type = AsyncTask::TaskType::LOAD;
    task.model_key = model_key;
    task.model_name = model_name;
    task.version = version;
    task.callback = callback;
    
    // Queue the task
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(task);
    }
    
    // Notify a worker thread
    queue_condition_.notify_one();
    
    return true;
}

/**
 * @brief Internal implementation of model loading
 */
bool InferenceManager::LoadModelInternal(const std::string& model_name, 
                                        const std::string& version,
                                        const std::string& model_key) {
    // Lock the models map
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    try {
        // Check if model is already loaded or in transition
        auto it = models_.find(model_key);
        if (it != models_.end()) {
            // Check current model state to determine action
            switch (it->second.state) {
                case ModelState::LOADED:
                    // Model already loaded - nothing to do
                    return true;
                    
                case ModelState::LOADING:
                    // Model is already being loaded - this is not an error for async requests
                    return true;
                    
                case ModelState::UNLOADING:
                    // Model is currently being unloaded - wait and retry would be better
                    SetError("Model is currently being unloaded: " + model_key);
                    return false;
                    
                case ModelState::ERROR:
                    // Model was in error state, allow attempt to reload
                    // Fall through to loading code
                    break;
                    
                default:
                    // Proceed with loading for other states
                    break;
            }
        } else {
            // Model doesn't exist in our map yet, create entry
            ModelInfo info;
            info.state = ModelState::UNAVAILABLE;  // Will update after checking repository
            info.state_changed_time = std::chrono::system_clock::now();
            models_.emplace(model_key, std::move(info));
            it = models_.find(model_key);
        }
        
        // Get model path and configuration from repository
        std::string resolved_version = version.empty() ? repository_->GetLatestVersion(model_name) : version;
        std::string model_path = repository_->GetModelPath(model_name, resolved_version);
        
        if (model_path.empty()) {
            it->second.state = ModelState::UNAVAILABLE;
            it->second.error_message = "Model not found in repository";
            it->second.state_changed_time = std::chrono::system_clock::now();
            SetError("Model not found: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
        
        // Update state to loading
        it->second.state = ModelState::LOADING;
        it->second.error_message.clear();
        it->second.state_changed_time = std::chrono::system_clock::now();
        
        // Get model configuration
        ModelConfig config = repository_->GetModelConfig(model_name, resolved_version);
        
        // Determine device type based on available hardware
        DeviceType device_type = cuda::IsCudaAvailable() ? DeviceType::GPU : DeviceType::CPU;
        int device_id = 0; // Use first device for now
        
        // Create model instance
        auto model = std::make_shared<Model>(model_path, config.type, config, device_type, device_id);
        it->second.model = model;
        
        // Perform actual model loading
        bool load_success = model->Load();
        if (!load_success) {
            // Update state to error if loading failed
            it->second.state = ModelState::ERROR;
            it->second.error_message = model->GetLastError();
            it->second.state_changed_time = std::chrono::system_clock::now();
            
            SetError("Failed to load model: " + model->GetLastError());
            return false;
        }
        
        // Successfully loaded - update state
        it->second.state = ModelState::LOADED;
        it->second.error_message.clear();
        it->second.state_changed_time = std::chrono::system_clock::now();
        
        return true;
    } catch (const std::exception& e) {
        // Update model state to error
        auto it = models_.find(model_key);
        if (it != models_.end()) {
            it->second.state = ModelState::ERROR;
            it->second.error_message = e.what();
            it->second.state_changed_time = std::chrono::system_clock::now();
        }
        
        SetError(std::string("Load model error: ") + e.what());
        return false;
    }
}

/**
 * @brief Unload a model synchronously
 */
bool InferenceManager::UnloadModel(const std::string& model_name, const std::string& version) {
    std::string model_key = MakeModelKey(model_name, version);
    return UnloadModelInternal(model_name, version, model_key);
}

/**
 * @brief Unload a model asynchronously
 */
bool InferenceManager::UnloadModelAsync(const std::string& model_name, 
                                       const std::string& version,
                                       ModelOperationCallback callback) {
    std::string model_key = MakeModelKey(model_name, version);
    
    // Create async task
    AsyncTask task;
    task.type = AsyncTask::TaskType::UNLOAD;
    task.model_key = model_key;
    task.model_name = model_name;
    task.version = version;
    task.callback = callback;
    
    // Queue the task
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        task_queue_.push(task);
    }
    
    // Notify a worker thread
    queue_condition_.notify_one();
    
    return true;
}

/**
 * @brief Internal implementation of model unloading
 */
bool InferenceManager::UnloadModelInternal(const std::string& model_name, 
                                          const std::string& version,
                                          const std::string& model_key) {
    // Lock the models map
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    try {
        // Find model
        auto it = models_.find(model_key);
        if (it == models_.end()) {
            // Model not found - consider this success since the end result is
            // that the model is not loaded, which is what was requested
            return true;
        }
        
        // Check model state to determine action
        switch (it->second.state) {
            case ModelState::UNLOADED:
            case ModelState::UNAVAILABLE:
                // Already unloaded or not available - nothing to do
                return true;
                
            case ModelState::UNLOADING:
                // Already being unloaded - this is not an error for async requests
                return true;
                
            case ModelState::LOADING:
                // Being loaded - can't unload yet
                SetError("Model is currently being loaded: " + model_key);
                return false;
                
            default:
                // Proceed with unloading for other states
                break;
        }
        
        // Update state to unloading
        it->second.state = ModelState::UNLOADING;
        it->second.error_message.clear();
        it->second.state_changed_time = std::chrono::system_clock::now();
        
        // Explicitly unload the model to release resources
        if (it->second.model) {
            it->second.model->Unload();
            it->second.model.reset(); // Release model resources
        }
        
        // Remove the model entry completely (alternative: keep with UNLOADED state)
        models_.erase(it);
        
        return true;
    } catch (const std::exception& e) {
        // If model exists, update its state
        auto it = models_.find(model_key);
        if (it != models_.end()) {
            it->second.state = ModelState::ERROR;
            it->second.error_message = e.what();
            it->second.state_changed_time = std::chrono::system_clock::now();
        }
        
        SetError(std::string("Unload model error: ") + e.what());
        return false;
    }
}

/**
 * @brief Check if a model is loaded and ready for inference
 */
bool InferenceManager::IsModelLoaded(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    // Only return true if model exists AND is in LOADED state
    return (it != models_.end() && it->second.state == ModelState::LOADED);
}

/**
 * @brief Get the current state of a model
 */
ModelState InferenceManager::GetModelState(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    if (it != models_.end()) {
        return it->second.state;
    }
    
    // Check if model exists in repository but isn't loaded
    if (repository_ && repository_->ModelExists(model_name, version)) {
        return ModelState::UNLOADED;
    }
    
    return ModelState::UNAVAILABLE;
}

/**
 * @brief Get detailed model status including error messages and timestamps
 */
std::string InferenceManager::GetModelStatus(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    std::stringstream json;
    json << "{\n";
    
    // Check if model is in our tracking map
    auto it = models_.find(model_key);
    if (it != models_.end()) {
        const auto& info = it->second;
        
        // Convert time_point to ISO string (simplified)
        auto time_t = std::chrono::system_clock::to_time_t(info.state_changed_time);
        std::string time_str = std::ctime(&time_t);
        if (!time_str.empty() && time_str.back() == '\n') {
            time_str.pop_back(); // Remove trailing newline
        }
        
        // Add basic model info
        json << "  \"name\": \"" << model_name << "\",\n";
        json << "  \"version\": \"" << (version.empty() ? repository_->GetLatestVersion(model_name) : version) << "\",\n";
        json << "  \"state\": \"" << ModelStateToString(info.state) << "\",\n";
        json << "  \"state_changed_time\": \"" << time_str << "\",\n";
        json << "  \"error_message\": \"" << EscapeJsonString(info.error_message) << "\"";
        
        // Add model-specific details if available
        if (info.model && info.state == ModelState::LOADED) {
            json << ",\n  \"type\": " << static_cast<int>(info.model->GetMetadata().type) << ",\n";
            json << "  \"memory_usage_bytes\": " << info.model->GetStats().memory_usage_bytes << ",\n";
            json << "  \"inference_count\": " << info.model->GetStats().inference_count;
        }
    } else if (repository_ && repository_->ModelExists(model_name, version)) {
        // Model exists in repository but isn't loaded
        json << "  \"name\": \"" << model_name << "\",\n";
        json << "  \"version\": \"" << (version.empty() ? repository_->GetLatestVersion(model_name) : version) << "\",\n";
        json << "  \"state\": \"UNLOADED\",\n";
        json << "  \"error_message\": \"\"";
    } else {
        // Model doesn't exist
        json << "  \"name\": \"" << model_name << "\",\n";
        json << "  \"version\": \"" << version << "\",\n";
        json << "  \"state\": \"UNAVAILABLE\",\n";
        json << "  \"error_message\": \"Model not found in repository\"";
    }
    
    json << "\n}";
    return json.str();
}

/**
 * @brief Helper to escape strings for JSON
 */
std::string EscapeJsonString(const std::string& input) {
    std::ostringstream ss;
    for (auto ch : input) {
        switch (ch) {
            case '\\': ss << "\\\\"; break;
            case '"': ss << "\\\""; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (ch < 32) {
                    // For control characters, use \uXXXX format
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                } else {
                    ss << ch;
                }
        }
    }
    return ss.str();
}

/**
 * @brief Helper to escape strings for JSON
 */
std::string EscapeJsonString(const std::string& input) {
    std::ostringstream ss;
    for (auto ch : input) {
        switch (ch) {
            case '\\': ss << "\\\\"; break;
            case '"': ss << "\\\""; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (ch < 32) {
                    // For control characters, use \uXXXX format
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch);
                } else {
                    ss << ch;
                }
        }
    }
    return ss.str();
}

/**
 * @brief List all available models in the repository
 */
std::vector<std::string> InferenceManager::ListModels() {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    if (!repository_) {
        return {};
    }
    
    return repository_->GetAvailableModels();
}

/**
 * @brief Get a loaded model by name and version
 */
std::shared_ptr<Model> InferenceManager::GetModel(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(models_mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    // Only return model if it exists AND is in LOADED state
    if (it == models_.end() || it->second.state != ModelState::LOADED) {
        return nullptr;
    }
    
    return it->second.model;
}

/**
 * @brief Run inference with a loaded model
 */
bool InferenceManager::RunInference(const std::string& model_name, 
                                   const std::string& version,
                                   const std::vector<Tensor>& inputs, 
                                   std::vector<Tensor>& outputs) {
    // Get the model (with lock)
    std::shared_ptr<Model> model;
    {
        std::lock_guard<std::mutex> lock(models_mutex_);
        
        std::string model_key = MakeModelKey(model_name, version);
        auto it = models_.find(model_key);
        if (it == models_.end()) {
            SetError("Model not found: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
        
        // Verify model is in LOADED state
        if (it->second.state != ModelState::LOADED) {
            SetError("Model not in loaded state: " + model_key + " (current state: " + 
                     ModelStateToString(it->second.state) + ")");
            return false;
        }
        
        model = it->second.model;
    }
    
    // Run inference without holding the lock (allows concurrent inference)
    bool success = model->Infer(inputs, outputs);
    if (!success) {
        SetError(model->GetLastError());
    }
    
    return success;
}

/**
 * @brief Get the last error message
 */
std::string InferenceManager::GetLastError() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    return last_error_;
}

/**
 * @brief Set error message and log it
 */
void InferenceManager::SetError(const std::string& error) const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    last_error_ = error;
    std::cerr << "InferenceManager error: " << error << std::endl;
}

} // namespace inference