package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/Oscar-W-Chen/gpu-ai-inference-server/inference_engine/binding"
	"github.com/gin-gonic/gin"
)

func main() {
	// Initialize logger
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("Starting AI Inference Server...")

	// Check CUDA availability directly from binding
	cudaAvailable := binding.IsCUDAAvailable()
	deviceCount := binding.GetDeviceCount()

	log.Printf("CUDA Available: %v", cudaAvailable)
	log.Printf("GPU Device Count: %v", deviceCount)

	// Print device info for each GPU
	if cudaAvailable && deviceCount > 0 {
		for i := 0; i < deviceCount; i++ {
			deviceInfo := binding.GetDeviceInfo(i)
			log.Printf("Device %d: %s", i, deviceInfo)
		}
	}

	// Initialize router
	router := gin.Default()

	// Define simple routes
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"time":   time.Now().Unix(),
		})
	})

	// Simple endpoint to get CUDA info
	router.GET("/cuda", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"cuda_available": cudaAvailable,
			"device_count":   deviceCount,
		})
	})

	// If CUDA is available, add more detailed device info
	if cudaAvailable {
		router.GET("/devices", func(c *gin.Context) {
			devices := make([]string, deviceCount)
			for i := 0; i < deviceCount; i++ {
				devices[i] = binding.GetDeviceInfo(i)
			}
			c.JSON(http.StatusOK, gin.H{
				"devices": devices,
			})
		})
	}

	// Initialize server
	serverPort := "8080"
	if port := os.Getenv("PORT"); port != "" {
		serverPort = port
	}

	// Start server in goroutine
	go func() {
		if err := router.Run(":" + serverPort); err != nil {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	log.Printf("Server listening on port %s", serverPort)

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("Shutting down server...")
	fmt.Println("Server exited")
}
