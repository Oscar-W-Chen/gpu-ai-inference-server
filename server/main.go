package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"syscall"
	"time"

	"github.com/Oscar-W-Chen/gpu-ai-inference-server/inference_engine/binding"
	"github.com/gin-gonic/gin"
	"github.com/gomarkdown/markdown"
	"golang.ngrok.com/ngrok"
	"golang.ngrok.com/ngrok/config"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
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
		// Note: You'll need a markdown parser. Let's use a simple one.
		// Add this to your imports: "github.com/gomarkdown/markdown"
		// And install it with: go get github.com/gomarkdown/markdown
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
            <title>AI Inference Server</title>
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
                <h2>API Documentation</h2>
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
	// Create inference manager with repository path
	repoPath := "./models"
	manager, err := binding.NewInferenceManager(repoPath)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer manager.Shutdown()

	// List available models
	modelNames := manager.ListModels()

	// Gather detailed information about each model
	modelDetails := make([]gin.H, 0, len(modelNames))
	for _, name := range modelNames {
		// Check if model is loaded
		isLoaded := manager.IsModelLoaded(name, "")

		// Build model details
		modelDetail := gin.H{
			"name":      name,
			"is_loaded": isLoaded,
			"state":     "AVAILABLE", // Would come from model states in a full implementation
		}

		modelDetails = append(modelDetails, modelDetail)
	}

	c.IndentedJSON(http.StatusOK, gin.H{
		"repository_path": repoPath,
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

	// Create repository path
	repoPath := "./models"

	// Debug logging
	log.Printf("Debug: Loading model '%s' version '%s' from repo '%s'", modelName, version, repoPath)

	// Check if model directory exists
	modelDir := filepath.Join(repoPath, modelName)
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		log.Printf("Debug: Model directory not found at '%s'", modelDir)
		c.IndentedJSON(http.StatusNotFound, gin.H{"error": "Model directory not found"})
		return
	}

	// Check model version directory and file
	var versionToUse string
	if version == "" {
		// Try to find latest version if not specified
		entries, err := os.ReadDir(modelDir)
		if err != nil || len(entries) == 0 {
			log.Printf("Debug: Cannot read model versions in '%s': %v", modelDir, err)
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
					log.Printf("Debug: Found model version: %s at '%s'", versionDir, possiblePath)
					versions = append(versions, versionDir)
				}
			}
		}

		if len(versions) == 0 {
			log.Printf("Debug: No valid model versions found in '%s'", modelDir)
			c.IndentedJSON(http.StatusNotFound, gin.H{"error": "No valid model versions found"})
			return
		}

		// Sort versions (simple string sort for now)
		sort.Strings(versions)
		versionToUse = versions[len(versions)-1] // Use last (highest) version
		log.Printf("Debug: Selected latest version: %s", versionToUse)
	} else {
		versionToUse = version
		modelVersionPath := filepath.Join(modelDir, versionToUse)
		modelPath := filepath.Join(modelVersionPath, "model.onnx")

		// Check if specified version exists
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			log.Printf("Debug: Model file not found at '%s'", modelPath)
			c.IndentedJSON(http.StatusNotFound, gin.H{
				"error": fmt.Sprintf("Model version '%s' not found", versionToUse),
			})
			return
		}
		log.Printf("Debug: Found specified model version at '%s'", modelPath)
	}

	// Create inference manager with repository path
	manager, err := binding.NewInferenceManager(repoPath)
	if err != nil {
		log.Printf("Debug: Failed to create inference manager: %v", err)
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer manager.Shutdown()

	// Check if model is already loaded
	if manager.IsModelLoaded(modelName, versionToUse) {
		log.Printf("Debug: Model '%s' version '%s' is already loaded", modelName, versionToUse)
		c.IndentedJSON(http.StatusCreated, gin.H{
			"message": "Model already loaded",
			"name":    modelName,
			"version": versionToUse,
		})
		return
	}

	// Load the model
	log.Printf("Debug: Attempting to load model '%s' version '%s'", modelName, versionToUse)
	err = manager.LoadModel(modelName, versionToUse)
	if err != nil {
		log.Printf("Debug: Failed to load model: %v", err)
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	log.Printf("Debug: Successfully loaded model '%s' version '%s'", modelName, versionToUse)
	c.IndentedJSON(http.StatusOK, gin.H{
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

	// Create inference manager with repository path
	repoPath := "./models"
	manager, err := binding.NewInferenceManager(repoPath)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer manager.Shutdown()

	// Check if model is loaded
	if !manager.IsModelLoaded(modelName, version) {
		c.IndentedJSON(http.StatusOK, gin.H{
			"message": "Model is not currently loaded",
			"name":    modelName,
			"version": version,
		})
		return
	}

	// Unload the model
	err = manager.UnloadModel(modelName, version)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.IndentedJSON(http.StatusCreated, gin.H{
		"message": "Model unloaded successfully",
		"name":    modelName,
		"version": version,
	})
}

// GetModelStatus gets detailed status information about a specific model
func GetModelStatus(c *gin.Context) {
	// Get model name from URL
	modelName := c.Param("name")

	// Get version from query parameter (optional)
	version := c.Query("version")

	// Create repository path
	repoPath := "./models"

	// Debug logging
	log.Printf("Debug: Getting status for model '%s' version '%s' from repo '%s'", modelName, version, repoPath)

	// Check if model directory exists
	modelDir := filepath.Join(repoPath, modelName)
	if _, err := os.Stat(modelDir); os.IsNotExist(err) {
		log.Printf("Debug: Model directory not found at '%s'", modelDir)
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

	// Create inference manager with repository path
	manager, err := binding.NewInferenceManager(repoPath)
	if err != nil {
		c.IndentedJSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	defer manager.Shutdown()

	// Check if model exists in repository
	models := manager.ListModels()
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

	isLoaded := manager.IsModelLoaded(modelName, versionToCheck)

	// Build model status
	modelStatus := gin.H{
		"name":               modelName,
		"version":            versionToCheck,
		"is_loaded":          isLoaded,
		"repository_path":    filepath.Join(repoPath, modelName),
		"available_versions": versions,
	}

	c.IndentedJSON(http.StatusOK, modelStatus)
}

func run(ctx context.Context) error {
	// Initialize logger
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting AI Inference Server...")

	// Check CUDA availability
	cudaAvailable := binding.IsCUDAAvailable()
	deviceCount := binding.GetDeviceCount()

	log.Printf("CUDA Available: %v", cudaAvailable)
	log.Printf("GPU Device Count: %v", deviceCount)

	if cudaAvailable && deviceCount > 0 {
		for i := 0; i < deviceCount; i++ {
			deviceInfo := binding.GetDeviceInfo(i)
			log.Printf("Device %d: %s", i, deviceInfo)
		}
	}

	// Initialize router
	router := gin.Default()
	router.GET("/", serveHome)
	router.GET("/health", getHealth)
	router.GET("/cuda", getCUDAInfo(cudaAvailable, deviceCount))
	router.GET("/models", GetModels)

	// Model management endpoints
	router.POST("/models/:name/load", LoadModel)
	router.POST("/models/:name/unload", UnloadModel)
	router.GET("/models/:name", GetModelStatus)

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

	log.Println("App URL:", listener.URL())

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
	fmt.Println("Server exited")
	return server.Shutdown(ctx)
}
