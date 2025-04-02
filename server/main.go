package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/Oscar-W-Chen/gpu-ai-inference-server/inference_engine/binding"
	"github.com/gin-gonic/gin"
	"github.com/gomarkdown/markdown"
	"golang.ngrok.com/ngrok"
	"golang.ngrok.com/ngrok/config"
)

// Global inference manager to avoid creating new ones for each request
var inferenceManager *binding.InferenceManager

// Application settings
const (
	ModelRepositoryPath = "./models"
	ServerLogPrefix     = "[SERVER] "
)

func main() {
	setupLogging()

	// Set Gin to release mode to disable debug output
	gin.SetMode(gin.ReleaseMode)

	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func setupLogging() {
	log.SetFlags(log.Ldate | log.Ltime)
	log.SetPrefix(ServerLogPrefix)
}

// serveHome serves the default HTML page with API docs
func serveHome(c *gin.Context) {
	// Try to read the API documentation markdown file
	apiDocsPath := "./docs/api.md"
	apiDocsHTML := ""

	// Read the markdown file
	mdContent, err := os.ReadFile(apiDocsPath)
	if err == nil {
		// Convert markdown to HTML
		htmlContent := markdown.ToHTML(mdContent, nil, nil)
		apiDocsHTML = string(htmlContent)
	} else {
		// If we can't read the file, provide a fallback
		apiDocsHTML = "<p>API documentation not available.</p>"
	}

	// Serve the HTML with embedded API docs
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPU AI Inference Server</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #333;
                }
                h1 {
                    border-bottom: 2px solid #eee;
                    padding-bottom: 10px;
                }
                pre {
                    background: #f4f4f4;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    overflow: auto;
                }
                code {
                    background: #f4f4f4;
                    padding: 2px 5px;
                    border-radius: 3px;
                }
                .container {
                    margin-top: 30px;
                }
            </style>
        </head>
        <body>
            <h1>Welcome to Oscar Chen's AI Inference Server!</h1>
            <h3>Leverage NVIDIA GPUs on Google Colab to load and run inference on your desired models</h3>
            
            <div class="container">
                <div class="api-docs">
                    `+apiDocsHTML+`
                </div>
            </div>
        </body>
        </html>
    `))
}

// getHealth returns the health status of the server
func getHealth(c *gin.Context) {
	c.IndentedJSON(http.StatusOK, gin.H{
		"status": "healthy",
		"time":   time.Now().Unix(),
	})
}

// getCUDAInfo returns if CUDA is available and the device counts
func getCUDAInfo(cudaAvailable bool, deviceCount int) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.IndentedJSON(http.StatusOK, gin.H{
			"cuda_available": cudaAvailable,
			"device_count":   deviceCount,
		})
	}
}

// getDevices gets individual device info
func getDevices(cudaAvailable bool, deviceCount int) gin.HandlerFunc {
	return func(c *gin.Context) {
		if cudaAvailable {
			devices := make([]string, deviceCount)
			for i := 0; i < deviceCount; i++ {
				devices[i] = binding.GetDeviceInfo(i)
			}
			c.IndentedJSON(http.StatusOK, gin.H{"devices": devices})
		} else {
			c.IndentedJSON(http.StatusOK, gin.H{"devices": []string{}})
		}
	}
}

// GetGPUMemory checks GPU memory usage, which is important for model management
func GetGPUMemory(cudaAvailable bool, deviceCount int) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !cudaAvailable {
			c.IndentedJSON(http.StatusServiceUnavailable, gin.H{"error": "CUDA not available"})
			return
		}

		memoryInfo := make([]gin.H, deviceCount)
		for i := 0; i < deviceCount; i++ {
			info, err := binding.GetMemoryInfo(i)
			if err != nil {
				c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			memoryInfo[i] = gin.H{
				"device_id":    i,
				"total_mb":     info.Total / (1024 * 1024),
				"free_mb":      info.Free / (1024 * 1024),
				"used_mb":      info.Used / (1024 * 1024),
				"used_percent": float64(info.Used) / float64(info.Total) * 100,
			}
		}

		c.IndentedJSON(http.StatusOK, gin.H{"memory_info": memoryInfo})
	}
}

// GetModels list the models in the repository directory
func GetModels(c *gin.Context) {
	// List available models
	modelNames := inferenceManager.ListModels()

	// Gather detailed information about each model
	modelDetails := make([]gin.H, 0, len(modelNames))
	for _, name := range modelNames {
		// Check if model is loaded
		isLoaded := inferenceManager.IsModelLoaded(name, "")

		// Build model details
		modelDetail := gin.H{
			"name":      name,
			"is_loaded": isLoaded,
			"state":     "AVAILABLE", // Would come from model states in a full implementation
		}

		modelDetails = append(modelDetails, modelDetail)
	}

	c.IndentedJSON(http.StatusOK, gin.H{
		"repository_path": ModelRepositoryPath,
		"model_count":     len(modelNames),
		"models":          modelDetails,
	})
}

// LoadModel loads a model from the repository
func LoadModel(c *gin.Context) {
	// Get model name from URL
	modelName := c.Param("name")

	// Get version from query parameter (optional)
	version := c.Query("version")

	// Check if model directory exists
	modelDir := filepath.Join(ModelRepositoryPath, modelName)
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		c.IndentedJSON(http.StatusNotFound, gin.H{"error": "Model directory not found"})
		return
	}

	// Check model version directory and file
	var versionToUse string
	if version == "" {
		// Try to find latest version if not specified
		entries, err := os.ReadDir(modelDir)
		if err != nil || len(entries) == 0 {
			c.IndentedJSON(http.StatusNotFound, gin.H{"error": "Cannot read model versions"})
			return
		}

		// Find directories that might be versions
		var versions []string
		for _, entry := range entries {
			if entry.IsDir() {
				versionDir := entry.Name()
				possiblePath := filepath.Join(modelDir, versionDir, "model.onnx")
				if _, err := os.Stat(possiblePath); err == nil {
					versions = append(versions, versionDir)
				}
			}
		}

		if len(versions) == 0 {
			c.IndentedJSON(http.StatusNotFound, gin.H{"error": "No valid model versions found"})
			return
		}

		// Sort versions (simple string sort for now)
		sort.Strings(versions)
		versionToUse = versions[len(versions)-1] // Use last (highest) version
	} else {
		versionToUse = version
		modelVersionPath := filepath.Join(modelDir, versionToUse)
		modelPath := filepath.Join(modelVersionPath, "model.onnx")

		// Check if specified version exists
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			c.IndentedJSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("Model version '%s' not found", versionToUse),
			})
			return
		}
	}

	// Check if model is already loaded
	if inferenceManager.IsModelLoaded(modelName, versionToUse) {
		log.Printf("Debug: Model '%s' version '%s' is already loaded", modelName, versionToUse)
		c.IndentedJSON(http.StatusOK, gin.H{
			"message": "Model already loaded",
			"name":    modelName,
			"version": versionToUse,
		})
		return
	}

	// Load the model
	err := inferenceManager.LoadModel(modelName, versionToUse)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.IndentedJSON(http.StatusAccepted, gin.H{
		"message": "Model loaded successfully",
		"name":    modelName,
		"version": versionToUse,
	})
}

// UnloadModel unloads a model from the server
func UnloadModel(c *gin.Context) {
	// Get model name from URL
	modelName := c.Param("name")

	// Get version from query parameter (optional)
	version := c.Query("version")

	// Find the actual version being used if version is empty
	versionToUse := version
	if versionToUse == "" {
		// Try to find the loaded version
		modelDir := filepath.Join(ModelRepositoryPath, modelName)
		entries, err := os.ReadDir(modelDir)
		if err == nil && len(entries) > 0 {
			// Find directories that might be versions
			var versions []string
			for _, entry := range entries {
				if entry.IsDir() {
					versionDir := entry.Name()
					possiblePath := filepath.Join(modelDir, versionDir, "model.onnx")
					if _, err := os.Stat(possiblePath); err == nil {
						// Check if this version is loaded
						if inferenceManager.IsModelLoaded(modelName, versionDir) {
							versions = append(versions, versionDir)
						}
					}
				}
			}

			if len(versions) > 0 {
				// Sort versions and use the latest
				sort.Strings(versions)
				versionToUse = versions[len(versions)-1]
			}
		}
	}

	// Check if model is loaded
	if !inferenceManager.IsModelLoaded(modelName, version) {
		c.IndentedJSON(http.StatusOK, gin.H{
			"message": "Model is not currently loaded",
			"name":    modelName,
			"version": versionToUse,
		})
		return
	}

	// Unload the model
	err := inferenceManager.UnloadModel(modelName, versionToUse)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.IndentedJSON(http.StatusCreated, gin.H{
		"message": "Model unloaded successfully",
		"name":    modelName,
		"version": versionToUse,
	})
}

// GetModelStatus gets detailed status information about a specific model
// GetModelStatus gets detailed status information about a specific model
func GetModelStatus(c *gin.Context) {
	// Get model name from URL
	modelName := c.Param("name")

	// Get version from query parameter (optional)
	version := c.Query("version")

	// Check if model directory exists
	modelDir := filepath.Join(ModelRepositoryPath, modelName)
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		c.IndentedJSON(http.StatusNotFound, gin.H{"error": "Model not found in repository"})
		return
	}

	// Find available versions
	versions := []string{}
	entries, err := os.ReadDir(modelDir)
	if err == nil {
		for _, entry := range entries {
			if entry.IsDir() {
				versionDir := entry.Name()
				possiblePath := filepath.Join(modelDir, versionDir, "model.onnx")
				if _, err := os.Stat(possiblePath); err == nil {
					versions = append(versions, versionDir)
				}
			}
		}
	}

	// Check if model exists in repository
	models := inferenceManager.ListModels()
	modelExists := false
	for _, name := range models {
		if name == modelName {
			modelExists = true
			break
		}
	}

	if !modelExists {
		c.IndentedJSON(http.StatusNotFound, gin.H{"error": "Model not found in repository"})
		return
	}

	// Check if model is loaded and get its status
	versionToCheck := version
	if versionToCheck == "" && len(versions) > 0 {
		// Sort versions
		sort.Strings(versions)
		versionToCheck = versions[len(versions)-1] // Latest version
	}

	isLoaded := inferenceManager.IsModelLoaded(modelName, versionToCheck)

	// Load model configuration from config.json
	var modelConfig *ModelConfig
	if versionToCheck != "" {
		modelConfig, err = loadModelConfig(modelName, versionToCheck)
		// If error, just log it but continue (config data will be nil)
		if err != nil {
			log.Printf("Warning: Could not load config for model %s: %v", modelName, err)
		}
	}

	// Build model status with configuration info if available
	modelStatus := gin.H{
		"name":               modelName,
		"version":            versionToCheck,
		"is_loaded":          isLoaded,
		"repository_path":    filepath.Join(ModelRepositoryPath, modelName),
		"available_versions": versions,
	}

	// Add configuration data if we successfully loaded it
	if modelConfig != nil {
		modelStatus["config"] = modelConfig
	}

	c.IndentedJSON(http.StatusOK, modelStatus)
}

//The following functions are related to the actual model inferencing
//===================================================================

// RunInference handles inference requests for a model
// RunInference handles inference requests for a model
func RunInference(c *gin.Context) {
	// Get model name from URL
	modelName := c.Param("name")

	// Get version from query parameter (optional)
	version := c.Query("version")

	log.Printf("Processing inference request for model: %s, version: %s", modelName, version)

	// Check if model is loaded
	if !inferenceManager.IsModelLoaded(modelName, version) {
		msg := fmt.Sprintf("Model '%s' is not loaded. Please load the model first.", modelName)
		log.Print(msg)
		c.JSON(http.StatusBadRequest, gin.H{
			"error": msg,
		})
		return
	}

	// Load model configuration from config.json
	modelConfig, err := loadModelConfig(modelName, version)
	if err != nil {
		log.Printf("Failed to load model configuration: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Failed to load model configuration: %v", err),
		})
		return
	}

	log.Printf("Model config loaded successfully: %+v", modelConfig)

	// Parse the request body - simplified to just expect input data
	var request struct {
		Inputs map[string]interface{} `json:"inputs"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		log.Printf("Invalid request format: %v", err)
		log.Printf("Request body: %s", c.Request.Body)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format: " + err.Error()})
		return
	}

	log.Printf("Received inputs: %+v", request.Inputs)

	if len(request.Inputs) == 0 {
		log.Print("No inputs provided in request")
		c.JSON(http.StatusBadRequest, gin.H{"error": "No inputs provided"})
		return
	}

	// Create tensor data objects based on model config and input data
	inputs := make([]binding.TensorData, 0, len(modelConfig.Inputs))
	for _, inputConfig := range modelConfig.Inputs {
		// Get input data from the request
		rawData, ok := request.Inputs[inputConfig.Name]
		if !ok {
			log.Printf("Required input '%s' not provided", inputConfig.Name)
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("Required input '%s' not provided", inputConfig.Name),
			})
			return
		}

		log.Printf("Processing input '%s' with data type '%s'", inputConfig.Name, inputConfig.DataType)
		log.Printf("Expected shape: %v", inputConfig.Shape)

		// Convert the input data to the right format based on data type - only care about float for now
		var data interface{}
		var dataType binding.DataType

		switch inputConfig.DataType {
		case "FLOAT32":
			dataType = binding.DataTypeFloat32

			// Try to convert input data to []float32
			floatData, err := convertToFloat32Array(rawData)
			if err != nil {
				log.Printf("Failed to convert input '%s' to float32 array: %v", inputConfig.Name, err)
				log.Printf("Raw data was: %+v", rawData)
				c.JSON(http.StatusBadRequest, gin.H{
					"error": fmt.Sprintf("Failed to convert input '%s' to float32 array: %v", inputConfig.Name, err),
				})
				return
			}

			log.Printf("Converted data: %+v (length: %d)", floatData, len(floatData))

			// Verify shape matches expected dims
			expectedElementCount := int64(1)
			for _, dim := range inputConfig.Shape {
				expectedElementCount *= dim
			}

			if int64(len(floatData)) != expectedElementCount {
				log.Printf("Data length mismatch: got %d elements, expected %d elements based on shape %v",
					len(floatData), expectedElementCount, inputConfig.Shape)
				c.JSON(http.StatusBadRequest, gin.H{
					"error": fmt.Sprintf("Input '%s' has wrong size: expected %d elements (shape %v), got %d",
						inputConfig.Name, expectedElementCount, inputConfig.Shape, len(floatData)),
				})
				return
			}

			data = floatData
		default:
			log.Printf("Unsupported data type '%s' for input '%s'", inputConfig.DataType, inputConfig.Name)
			c.JSON(http.StatusBadRequest, gin.H{
				"error": fmt.Sprintf("Unsupported data type '%s' for input '%s'", inputConfig.DataType, inputConfig.Name),
			})
			return
		}

		// Create tensor with data and shape from config
		tensor := binding.TensorData{
			Name:     inputConfig.Name,
			DataType: dataType,
			Shape:    binding.Shape{Dims: inputConfig.Shape},
			Data:     data,
		}

		log.Printf("Created tensor for input '%s' with shape %v", tensor.Name, tensor.Shape.Dims)

		inputs = append(inputs, tensor)
	}

	log.Printf("Prepared %d tensor inputs, calling inference engine", len(inputs))

	// Call the binding layer to run inference
	outputs, err := inferenceManager.RunInference(modelName, version, inputs)
	if err != nil {
		log.Printf("Inference failed: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("Inference failed: %v", err),
		})
		return
	}

	log.Printf("Inference succeeded, processing %d outputs", len(outputs))

	// Convert the output tensors to a more user-friendly response
	// that includes classification labels if available
	responseOutputs := processOutputs(outputs, modelConfig.Outputs)

	// Send the response
	c.IndentedJSON(http.StatusOK, gin.H{
		"model_name":    modelName,
		"model_version": version,
		"outputs":       responseOutputs,
	})

	log.Printf("Inference response sent successfully")
}

// Model configuration structure
type ModelConfig struct {
	Name    string         `json:"name"`
	Version string         `json:"version"`
	Inputs  []InputConfig  `json:"inputs"`
	Outputs []OutputConfig `json:"outputs"`
}

type InputConfig struct {
	Name     string  `json:"name"`
	Dims     []int64 `json:"dims"`
	Shape    []int64 `json:"shape"`
	DataType string  `json:"data_type"`
}

type OutputConfig struct {
	Name          string  `json:"name"`
	Dims          []int64 `json:"dims"`
	Shape         []int64 `json:"shape"`
	DataType      string  `json:"data_type"`
	LabelFilename string  `json:"label_filename,omitempty"`
}

// Load model configuration from config.json
func loadModelConfig(modelName, version string) (*ModelConfig, error) {
	modelPath := filepath.Join(ModelRepositoryPath, modelName)

	// If version is provided, use it, otherwise find the latest
	if version != "" {
		modelPath = filepath.Join(modelPath, version)
	} else {
		// Find the latest version
		entries, err := os.ReadDir(modelPath)
		if err != nil {
			return nil, err
		}

		var versions []string
		for _, entry := range entries {
			if entry.IsDir() && isNumeric(entry.Name()) {
				versions = append(versions, entry.Name())
			}
		}

		if len(versions) == 0 {
			return nil, fmt.Errorf("no version directories found for model '%s'", modelName)
		}

		sort.Strings(versions)
		latestVersion := versions[len(versions)-1]
		modelPath = filepath.Join(modelPath, latestVersion)
	}

	// Read config.json
	configPath := filepath.Join(modelPath, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	// Parse the config.json
	var config ModelConfig
	if err := json.Unmarshal(configData, &config); err != nil {
		return nil, err
	}

	return &config, nil
}

// Process output tensors and include classification labels if available
func processOutputs(outputs []binding.TensorData, outputConfigs []OutputConfig) []map[string]interface{} {
	responseOutputs := make([]map[string]interface{}, 0, len(outputs))

	// Create a map of output name to output config
	outputConfigMap := make(map[string]OutputConfig)
	for _, config := range outputConfigs {
		outputConfigMap[config.Name] = config
	}

	for _, output := range outputs {
		outputMap := map[string]interface{}{
			"name":      output.Name,
			"data_type": output.DataType,
			"shape":     output.Shape.Dims,
			"data":      output.Data,
		}

		// If this output has labels, try to load them
		if config, ok := outputConfigMap[output.Name]; ok && config.LabelFilename != "" {
			// Try to load labels file
			labels, err := loadLabelFile(output.Name, config.LabelFilename)
			if err == nil && len(labels) > 0 {
				// If we have labels, add top classification results
				if floatData, ok := output.Data.([]float32); ok {
					// Find top classes
					topClasses := findTopClasses(floatData, labels, 5)
					outputMap["classifications"] = topClasses
				}
			}
		}

		responseOutputs = append(responseOutputs, outputMap)
	}

	return responseOutputs
}

// Load label file for classification outputs
func loadLabelFile(outputName, labelFilename string) ([]string, error) {
	// Try to find the label file in the model directory
	labelPath := filepath.Join(ModelRepositoryPath, labelFilename)

	// Read the label file
	labelData, err := os.ReadFile(labelPath)
	if err != nil {
		return nil, err
	}

	// Parse the labels (one per line)
	labels := strings.Split(string(labelData), "\n")

	// Trim whitespace and remove empty lines
	var cleanLabels []string
	for _, label := range labels {
		label = strings.TrimSpace(label)
		if label != "" {
			cleanLabels = append(cleanLabels, label)
		}
	}

	return cleanLabels, nil
}

// Find top N classes from classification results
func findTopClasses(probabilities []float32, labels []string, topN int) []map[string]interface{} {
	// Create pairs of (index, probability)
	type indexProb struct {
		Index int
		Prob  float32
	}

	pairs := make([]indexProb, 0, len(probabilities))
	for i, prob := range probabilities {
		pairs = append(pairs, indexProb{Index: i, Prob: prob})
	}

	// Sort by probability (descending)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Prob > pairs[j].Prob
	})

	// Take top N
	if topN > len(pairs) {
		topN = len(pairs)
	}

	// Convert to output format
	results := make([]map[string]interface{}, 0, topN)
	for i := 0; i < topN; i++ {
		index := pairs[i].Index
		prob := pairs[i].Prob

		result := map[string]interface{}{
			"index":       index,
			"probability": prob,
		}

		// Add label if available
		if index < len(labels) {
			result["label"] = labels[index]
		}

		results = append(results, result)
	}

	return results
}

// Helper function to check if a string is numeric
func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// Helper functions to convert input data
// Convert interface{} to []float32
func convertToFloat32Array(data interface{}) ([]float32, error) {
	// Marshal and unmarshal to convert various number formats to float32
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	var floatArray []float32
	if err := json.Unmarshal(jsonData, &floatArray); err != nil {
		return nil, err
	}

	return floatArray, nil
}

//===========================================================

// creates a singleton inference manager
func InitializeInferenceManager() error {
	var err error

	// Create model repository directory if it doesn't exist
	if _, err := os.Stat(ModelRepositoryPath); os.IsNotExist(err) {
		if err := os.MkdirAll(ModelRepositoryPath, 0755); err != nil {
			return fmt.Errorf("failed to create model repository: %v", err)
		}
	}

	// Initialize the inference manager
	inferenceManager, err = binding.NewInferenceManager(ModelRepositoryPath)
	if err != nil {
		return fmt.Errorf("failed to creating inference manager: %v", err)
	}
	return nil
}

func run(ctx context.Context) error {
	log.Println("Starting AI Inference Server...")

	// Check CUDA availability
	cudaAvailable := binding.IsCUDAAvailable()
	deviceCount := binding.GetDeviceCount()

	log.Printf("CUDA Available: %v, GPU Device Count: %v", cudaAvailable, deviceCount)

	if cudaAvailable && deviceCount > 0 {
		for i := 0; i < deviceCount; i++ {
			log.Printf("Device %d: %s", i, binding.GetDeviceInfo(i))
		}
	}

	// Initialize the global inference manager
	if err := InitializeInferenceManager(); err != nil {
		return err
	}
	defer inferenceManager.Shutdown()

	// Initialize router with default logger and recovery middleware
	router := gin.New()
	router.Use(gin.Recovery())

	// Use a custom logger that is more concise
	router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		// Return a minimal log format that only shows when there's an error
		if param.StatusCode >= 400 {
			return fmt.Sprintf("[SERVER] %v | %3d | %s | %s\n",
				param.TimeStamp.Format("2006/01/02 15:04:05"),
				param.StatusCode,
				param.Method,
				param.Path,
			)
		}
		return ""
	}))

	// Define routes
	router.GET("/", serveHome)
	router.GET("/health", getHealth)
	router.GET("/cuda", getCUDAInfo(cudaAvailable, deviceCount))
	router.GET("/models", GetModels)

	// Model management endpoints
	router.POST("/models/:name/load", LoadModel)
	router.POST("/models/:name/unload", UnloadModel)
	router.GET("/models/:name", GetModelStatus)
	router.POST("/models/:name/infer", RunInference)

	if cudaAvailable {
		router.GET("/devices", getDevices(cudaAvailable, deviceCount))
		router.GET("gpu/memory", GetGPUMemory(cudaAvailable, deviceCount))
	}

	// Start ngrok listener
	listener, err := ngrok.Listen(ctx,
		config.HTTPEndpoint(),
		ngrok.WithAuthtokenFromEnv(),
	)
	if err != nil {
		return err
	}

	log.Println("Server URL:", listener.URL())

	// Run the server in a goroutine
	server := &http.Server{Handler: router}
	go func() {
		if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Graceful shutdown handling
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")

	// Create a timeout context for shutdown
	shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	if err := server.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("server forced to shutdown: %v", err)
	}

	return nil
}
