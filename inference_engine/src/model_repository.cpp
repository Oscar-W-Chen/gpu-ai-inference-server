#include "model_repository.h"
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

namespace inference {

ModelRepository::ModelRepository(const std::string& repository_path)
    : repository_path_(repository_path) {
    // Ensure path exists
    if (!std::filesystem::exists(repository_path)) {
        std::cerr << "Warning: Model repository path does not exist: " << repository_path << std::endl;
        // Create the directory if it doesn't exist
        std::filesystem::create_directories(repository_path);
    }
}

bool ModelRepository::ScanRepository() {
    model_versions_.clear();
    
    try {
        if (!std::filesystem::exists(repository_path_)) {
            std::cerr << "Model repository path does not exist: " << repository_path_ << std::endl;
            return false;
        }

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

std::vector<std::string> ModelRepository::GetAvailableModels() const {
    std::vector<std::string> models;
    for (const auto& model : model_versions_) {
        models.push_back(model.first);
    }
    return models;
}

bool ModelRepository::ModelExists(const std::string& model_name, const std::string& version) const {
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

std::string ModelRepository::GetModelPath(const std::string& model_name, const std::string& version) const {
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

ModelConfig ModelRepository::GetModelConfig(const std::string& model_name, const std::string& version) const {
    ModelConfig config;
    std::string model_path = GetModelPath(model_name, version);
    
    if (model_path.empty()) {
        return config;
    }

    // Set basic model properties
    config.name = model_name;
    config.version = version.empty() ? GetLatestVersion(model_name) : version;
    
    // Try to load config file
    std::string config_file_path = model_path + "/config.json";
    std::ifstream config_file(config_file_path);
    
    if (config_file.is_open()) {
        // Parse JSON config
        try {
            std::stringstream buffer;
            buffer << config_file.rdbuf();
            // In a real implementation, parse JSON here
            // For now, just check if the file has content
            if (buffer.str().length() > 0) {
                // Simplified parsing - in a real implementation use a proper JSON parser
                // This is just for detecting that the file exists and has content
                
                // Try to detect model type based on files
                config.type = DetectModelType(model_path);
                
                // Add default inputs and outputs
                config.input_names = {"input"};
                config.output_names = {"output"};
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing config file: " << e.what() << std::endl;
        }
    } else {
        // No config file, try to infer config from model file
        config.type = DetectModelType(model_path);
        
        // Default values for inputs and outputs
        config.input_names = {"input"};
        config.output_names = {"output"};
    }

    return config;
}

ModelType ModelRepository::DetectModelType(const std::filesystem::path& model_path) const {
    // Check for different model file types
    if (std::filesystem::exists(model_path / "model.onnx")) {
        return ModelType::ONNX;
    }
    else if (std::filesystem::exists(model_path / "saved_model.pb")) {
        return ModelType::TENSORFLOW;
    }
    else if (std::filesystem::exists(model_path / "model.plan")) {
        return ModelType::TENSORRT;
    }
    else if (std::filesystem::exists(model_path / "model.pt")) {
        return ModelType::PYTORCH;
    }
    
    // Default if unable to determine
    return ModelType::UNKNOWN;
}

std::string ModelRepository::GetLatestVersion(const std::string& model_name) const {
    auto it = model_versions_.find(model_name);
    if (it == model_versions_.end() || it->second.empty()) {
        return "";
    }
    // First version in the list is the latest (due to sorting in ScanRepository)
    return it->second.front();
}

std::vector<std::string> ModelRepository::GetModelVersions(const std::string& model_name) const {
    auto it = model_versions_.find(model_name);
    if (it == model_versions_.end()) {
        return {};
    }
    return it->second;
}

bool ModelRepository::HasModelConfig(const std::filesystem::path& model_path) const {
    // Check for common model files or configuration
    return std::filesystem::exists(model_path / "config.json") ||
           std::filesystem::exists(model_path / "model.onnx") ||
           std::filesystem::exists(model_path / "model.pt") ||
           std::filesystem::exists(model_path / "saved_model.pb") ||
           std::filesystem::exists(model_path / "model.plan");
}

} // namespace inference