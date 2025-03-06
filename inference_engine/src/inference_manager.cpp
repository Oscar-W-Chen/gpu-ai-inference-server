#include "inference_manager.h"
#include "cuda_utils.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <chrono>

namespace inference {

//==============================================================================
// ModelRepository Class Implementation
//==============================================================================
/**
 * @brief Manages access to models stored in the filesystem
 * 
 * The ModelRepository handles scanning, versioning, and providing access to
 * model files stored in a repository directory structure similar to NVIDIA Triton.
 * Structure expected:
 *   repository_path/
 *     model_name1/
 *       1/          # version directory
 *         config.pbtxt or model files
 *       2/
 *         config.pbtxt or model files
 *     model_name2/
 *       ...
 */
class ModelRepository {
public:
    /**
     * @brief Constructor with repository path
     * @param path Path to model repository
     */
    ModelRepository(const std::string& path) : repository_path_(path) {}

    /**
     * @brief Scan the repository for models and versions
     * @return true if scan was successful, false otherwise
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
                            return std::stoi(a) > std::stoi(b);
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
     * @brief Get list of all available model names
     * @return Vector of model names
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
     * @param model_name Model name
     * @param version Model version (empty for latest)
     * @return Full path to the model directory, or empty string if not found
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
     * @param model_name Model name
     * @param version Model version (empty for latest)
     * @return Model configuration object
     * 
     * In a real implementation, this would parse a config file.
     * This simplified version just creates a basic configuration.
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
     * @param model_name Model name
     * @return Latest version number or empty string if model not found
     */
    std::string GetLatestVersion(const std::string& model_name) const {
        auto it = model_versions_.find(model_name);
        if (it == model_versions_.end() || it->second.empty()) {
            return "";
        }
        // First version in the list is the latest (due to sorting in ScanRepository)
        return it->second.front();
    }

private:
    // Path to model repository directory
    std::string repository_path_;
    
    // Map of model names to their available versions (sorted by version)
    std::unordered_map<std::string, std::vector<std::string>> model_versions_;

    /**
     * @brief Check if a directory contains valid model files
     * @param model_path Path to check
     * @return true if valid model files exist
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
     * @param model_path Path to model directory
     * @return Detected model type or UNKNOWN
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
// Helper Function for Model State
//==============================================================================

/**
 * @brief Convert model state enum to string representation
 * @param state Model state
 * @return String representation of state
 */
std::string StateToString(ModelState state) {
    switch (state) {
        case ModelState::UNLOADED: return "UNLOADED";
        case ModelState::LOADING: return "LOADING";
        case ModelState::LOADED: return "LOADED";
        case ModelState::UNLOADING: return "UNLOADING";
        case ModelState::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

//==============================================================================
// InferenceManager Implementation
//==============================================================================

/**
 * @brief Constructor
 * @param model_repository_path Path to model repository
 */
InferenceManager::InferenceManager(const std::string& model_repository_path)
    : model_repository_path_(model_repository_path) {
}

/**
 * @brief Destructor, ensures all resources are cleaned up
 */
InferenceManager::~InferenceManager() {
    Shutdown();
}

/**
 * @brief Initialize the inference manager and repository
 * @return true if initialization successful
 */
bool InferenceManager::Initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
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
 * 
 * Unloads all models and frees resources
 */
void InferenceManager::Shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Unload all models
    models_.clear();
    
    // Clear repository
    repository_.reset();
}

/**
 * @brief Load a model into memory
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @return true if model loaded successfully
 * 
 * Loads a model from the repository into memory.
 * Thread-safe: uses mutex to prevent concurrent modifications.
 */
bool InferenceManager::LoadModel(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Check initialization
        if (!repository_) {
            SetError("Inference manager not initialized");
            return false;
        }
        
        // Check if model is already loaded or in transition
        std::string model_key = MakeModelKey(model_name, version);
        auto it = models_.find(model_key);
        if (it != models_.end()) {
            // Check current model state to determine action
            switch (it->second.state) {
                case ModelState::LOADED:
                    // Model already loaded - nothing to do
                    return true;
                case ModelState::LOADING:
                    SetError("Model is already being loaded: " + model_key);
                    return false;
                case ModelState::UNLOADING:
                    SetError("Model is currently being unloaded: " + model_key);
                    return false;
                case ModelState::ERROR:
                    // Model was in error state, allow attempt to reload
                    break;
                default:
                    // Model exists but is unloaded, proceed with loading
                    break;
            }
        }
        
        // Get model path and configuration from repository
        std::string resolved_version = version.empty() ? repository_->GetLatestVersion(model_name) : version;
        std::string model_path = repository_->GetModelPath(model_name, resolved_version);
        
        if (model_path.empty()) {
            SetError("Model not found: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
        
        ModelConfig config = repository_->GetModelConfig(model_name, resolved_version);
        
        // Determine device type based on available hardware
        // Use GPU if available, otherwise fall back to CPU
        DeviceType device_type = cuda::IsCudaAvailable() ? DeviceType::GPU : DeviceType::CPU;
        int device_id = 0; // Use first device for now
        
        // Create model instance
        auto model = std::make_shared<Model>(model_path, config.type, config, device_type, device_id);
        
        // Update state to loading and track the time
        ModelInfo model_info;
        model_info.model = model;
        model_info.state = ModelState::LOADING;
        model_info.state_changed_time = std::chrono::system_clock::now();
        models_[model_key] = model_info;
        
        // Perform actual model loading
        if (!model->Load()) {
            // Update state to error if loading failed
            models_[model_key].state = ModelState::ERROR;
            models_[model_key].error_message = model->GetLastError();
            models_[model_key].state_changed_time = std::chrono::system_clock::now();
            
            SetError("Failed to load model: " + model->GetLastError());
            return false;
        }
        
        // Successfully loaded - update state
        models_[model_key].state = ModelState::LOADED;
        models_[model_key].state_changed_time = std::chrono::system_clock::now();
        
        return true;
    } catch (const std::exception& e) {
        SetError(std::string("Load model error: ") + e.what());
        return false;
    }
}

/**
 * @brief Unload a model from memory
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @return true if model unloaded successfully
 * 
 * Unloads a model from memory and releases resources.
 * Thread-safe: uses mutex to prevent concurrent modifications.
 */
bool InferenceManager::UnloadModel(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Get model key
        std::string model_key = MakeModelKey(model_name, version);
        
        // Find model
        auto it = models_.find(model_key);
        if (it == models_.end()) {
            SetError("Model not found: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
        
        // Check model state to determine action
        switch (it->second.state) {
            case ModelState::UNLOADED:
                // Already unloaded - nothing to do
                return true;
            case ModelState::UNLOADING:
                SetError("Model is already being unloaded: " + model_key);
                return false;
            case ModelState::LOADING:
                SetError("Model is currently being loaded: " + model_key);
                return false;
            default:
                // Proceed with unloading for other states
                break;
        }
        
        // Update state to unloading
        it->second.state = ModelState::UNLOADING;
        it->second.state_changed_time = std::chrono::system_clock::now();
        
        // Explicitly unload the model to release resources
        if (it->second.model) {
            it->second.model->Unload();
        }
        
        // Update state to unloaded
        it->second.state = ModelState::UNLOADED;
        it->second.state_changed_time = std::chrono::system_clock::now();
        
        // Remove the model completely from the map
        // Alternative: keep entry but set model=nullptr if we want history
        models_.erase(it);
        
        return true;
    } catch (const std::exception& e) {
        SetError(std::string("Unload model error: ") + e.what());
        return false;
    }
}

/**
 * @brief Check if a model is loaded and ready for inference
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @return true if model is loaded, false otherwise
 */
bool InferenceManager::IsModelLoaded(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    // Only return true if model exists AND is in LOADED state
    return (it != models_.end() && it->second.state == ModelState::LOADED);
}

/**
 * @brief Get the current state of a model
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @return Current state of the model (UNLOADED if not found)
 */
ModelState InferenceManager::GetModelState(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    if (it != models_.end()) {
        return it->second.state;
    }
    return ModelState::UNLOADED;
}

/**
 * @brief List all available models in the repository
 * @return Vector of model names
 */
std::vector<std::string> InferenceManager::ListModels() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!repository_) {
        return {};
    }
    
    return repository_->GetAvailableModels();
}

/**
 * @brief Get a loaded model by name and version
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @return Shared pointer to model, or nullptr if not found or not loaded
 */
std::shared_ptr<Model> InferenceManager::GetModel(const std::string& model_name, const std::string& version) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string model_key = MakeModelKey(model_name, version);
    auto it = models_.find(model_key);
    // Only return model if it exists AND is in LOADED state
    if (it == models_.end() || it->second.state != ModelState::LOADED) {
        return nullptr;
    }
    
    return it->second.model;
}

/**
 * @brief Execute inference on a loaded model
 * @param model_name Model name
 * @param version Model version (empty for latest)
 * @param inputs Input tensors
 * @param outputs Output tensors to be filled
 * @return true if inference successful
 * 
 * Thread safety: Uses a two-phase approach to allow concurrent inferences
 * while protecting the model map.
 */
bool InferenceManager::RunInference(const std::string& model_name, 
                                   const std::string& version,
                                   const std::vector<Tensor>& inputs, 
                                   std::vector<Tensor>& outputs) {
    // Phase 1: Get the model under lock
    std::shared_ptr<Model> model;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::string model_key = MakeModelKey(model_name, version);
        auto it = models_.find(model_key);
        if (it == models_.end()) {
            SetError("Model not found: " + model_name + (version.empty() ? "" : ":" + version));
            return false;
        }
        
        // Verify model is in LOADED state
        if (it->second.state != ModelState::LOADED) {
            SetError("Model not in loaded state: " + model_key + " (current state: " + 
                     StateToString(it->second.state) + ")");
            return false;
        }
        
        model = it->second.model;
    }
    
    // Phase 2: Run inference without holding the lock
    // This allows multiple inferences to run concurrently
    bool success = model->Infer(inputs, outputs);
    if (!success) {
        SetError(model->GetLastError());
    }
    
    return success;
}

/**
 * @brief Get the last error message
 * @return Last error message string
 */
std::string InferenceManager::GetLastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

/**
 * @brief Create a unique key for a model+version combination
 * @param name Model name
 * @param version Model version
 * @return Unique key string in format "name:version"
 * 
 * If version is empty, attempts to get the latest version.
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
 * @brief Set error message and log it
 * @param error Error message
 */
void InferenceManager::SetError(const std::string& error) const {
    last_error_ = error;
    std::cerr << "InferenceManager error: " << error << std::endl;
}

} // namespace inference