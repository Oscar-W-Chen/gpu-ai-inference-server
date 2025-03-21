package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Oscar-W-Chen/gpu-ai-inference-server/inference_engine/binding"
	"github.com/gin-gonic/gin"
	"golang.ngrok.com/ngrok"
	"golang.ngrok.com/ngrok/config"
)

func main() {
	if err := run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

// serveHome serves the default HTML page
func serveHome(c *gin.Context) {
	c.Data(http.StatusOK, "text/html; charset=utf-8", []byte(`
		<!DOCTYPE html>
		<html>
		<head>
			<title>AI Inference Server</title>
		</head>
		<body>
			<h1>Welcome to Oscar Chen's AI Inference Server!</h1>
			<h3>Leverage NVIDIA GPUs on Google Colab to load and run inference on your desired models</h3>
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
	models := manager.ListModels()

	c.IndentedJSON(http.StatusOK, gin.H{
		"repository_path": repoPath,
		"models":          models,
	})
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
