#ifndef MODEL_REPOSITORY_H
#define MODEL_REPOSITORY_H

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include "model.h"

namespace inference {

/**
 * @class ModelRepository
 * @brief Manages access to models stored in the filesystem
 * 
 * The ModelRepository class is responsible for scanning a directory structure
 * for models, tracking their versions, and providing access to model metadata.
 */
class ModelRepository {
public:
    /**
     * @brief Constructor with repository path
     * @param repository_path Path to the model repository directory
     */
    ModelRepository(const std::string& repository_path);
    
    /**
     * @brief Scan the repository for models and versions
     * @return true if scan was successful, false if error
     */
    bool ScanRepository();
    
    /**
     * @brief Get a list of all available model names
     * @return Vector of model names
     */
    std::vector<std::string> GetAvailableModels() const;
    
    /**
     * @brief Check if a model exists in the repository
     * @param model_name Name of the model to check
     * @param version Version of the model to check (empty for any version)
     * @return true if model exists, false otherwise
     */
    bool ModelExists(const std::string& model_name, const std::string& version = "") const;
    
    /**
     * @brief Get the filesystem path to a model
     * @param model_name Name of the model
     * @param version Version of the model (empty for latest)
     * @return Filesystem path to the model, empty if not found
     */
    std::string GetModelPath(const std::string& model_name, const std::string& version = "") const;
    
    /**
     * @brief Get configuration for a model
     * @param model_name Name of the model
     * @param version Version of the model (empty for latest)
     * @return ModelConfig with settings for the model
     */
    ModelConfig GetModelConfig(const std::string& model_name, const std::string& version = "") const;
    
    /**
     * @brief Get the latest version for a model
     * @param model_name Name of the model
     * @return Version string, empty if model not found
     */
    std::string GetLatestVersion(const std::string& model_name) const;
    
    /**
     * @brief Get all available versions for a model
     * @param model_name Name of the model
     * @return Vector of version strings
     */
    std::vector<std::string> GetModelVersions(const std::string& model_name) const;
    
private:
    // Path to model repository directory
    std::string repository_path_;
    
    // Map of model names to their available versions (sorted by version)
    std::unordered_map<std::string, std::vector<std::string>> model_versions_;
    
    /**
     * @brief Check if a directory contains valid model files
     * @param model_path Path to check for model files
     * @return true if valid model files found, false otherwise
     */
    bool HasModelConfig(const std::filesystem::path& model_path) const;
    
    /**
     * @brief Attempt to detect model type from files in directory
     * @param model_path Path to model directory
     * @return Detected ModelType, or UNKNOWN if can't determine
     */
    ModelType DetectModelType(const std::filesystem::path& model_path) const;
};

} // namespace inference

#endif // MODEL_REPOSITORY_H