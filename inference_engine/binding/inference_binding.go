package binding

/*
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../../build/inference_engine -linference_engine -lstdc++ -Wl,-rpath,${SRCDIR}/../../build/inference_engine
#include <stdlib.h>
#include "inference_bridge.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// DataType represents tensor data types
type DataType int

const (
	DataTypeFloat32 DataType = C.DATATYPE_FLOAT32
	DataTypeInt32   DataType = C.DATATYPE_INT32
	DataTypeInt64   DataType = C.DATATYPE_INT64
	DataTypeUint8   DataType = C.DATATYPE_UINT8
	DataTypeInt8    DataType = C.DATATYPE_INT8
	DataTypeString  DataType = C.DATATYPE_STRING
	DataTypeBool    DataType = C.DATATYPE_BOOL
	DataTypeFP16    DataType = C.DATATYPE_FP16
	DataTypeUnknown DataType = C.DATATYPE_UNKNOWN
)

// DeviceType represents device types for model execution
type DeviceType int

const (
	DeviceCPU DeviceType = C.DEVICE_CPU
	DeviceGPU DeviceType = C.DEVICE_GPU
)

// ModelType represents supported model frameworks
type ModelType int

const (
	ModelUnknown    ModelType = C.MODEL_UNKNOWN
	ModelTensorFlow ModelType = C.MODEL_TENSORFLOW
	ModelTensorRT   ModelType = C.MODEL_TENSORRT
	ModelONNX       ModelType = C.MODEL_ONNX
	ModelPyTorch    ModelType = C.MODEL_PYTORCH
	ModelCustom     ModelType = C.MODEL_CUSTOM
)

// Shape represents the dimensions of a tensor
type Shape struct {
	Dims []int64
}

// TensorData represents the data for a tensor
type TensorData struct {
	Name     string
	DataType DataType
	Shape    Shape
	Data     interface{} // Will hold Go slice of appropriate type
}

// ModelConfig represents configuration for a model
type ModelConfig struct {
	Name            string
	Version         string
	Type            ModelType
	MaxBatchSize    int
	InputNames      []string
	OutputNames     []string
	InstanceCount   int
	DynamicBatching bool
}

// ModelMetadata represents metadata about a model
type ModelMetadata struct {
	Name        string
	Version     string
	Type        ModelType
	Inputs      []string
	Outputs     []string
	Description string
	LoadTimeNs  int64
}

// ModelStats represents statistics for a model
type ModelStats struct {
	InferenceCount       int64
	TotalInferenceTimeNs int64
	LastInferenceTimeNs  int64
	MemoryUsageBytes     uint64
}

// InferenceManager handles model management
type InferenceManager struct {
	handle C.InferenceManagerHandle
	// Track loaded models with a map
	loadedModels      map[string]*Model
	loadedModelsMutex sync.RWMutex
}

// Model represents a loaded model
type Model struct {
	handle  C.ModelHandle
	name    string
	version string
}

// Tensor represents a tensor for inference
type Tensor struct {
	handle C.TensorHandle
}

type MemoryInfo struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// CUDA utility functions

// IsCUDAAvailable checks if CUDA is available on the system
func IsCUDAAvailable() bool {
	return bool(C.IsCudaAvailable())
}

// GetDeviceCount returns the number of CUDA devices
func GetDeviceCount() int {
	return int(C.GetDeviceCount())
}

// GetDeviceInfo returns information about a CUDA device
func GetDeviceInfo(deviceID int) string {
	cInfo := C.GetDeviceInfo(C.int(deviceID))
	defer C.free(unsafe.Pointer(cInfo))
	return C.GoString(cInfo)
}

// GetMemoryInfo returns memory information for a specific GPU device
func GetMemoryInfo(deviceID int) (MemoryInfo, error) {
	var info MemoryInfo

	if !IsCUDAAvailable() {
		return info, errors.New("CUDA is not available")
	}

	cInfo := C.GetMemoryInfo(C.int(deviceID))

	info.Total = uint64(cInfo.total)
	info.Free = uint64(cInfo.free)
	info.Used = uint64(cInfo.used)

	// Verify we got valid data
	if info.Total == 0 {
		return info, errors.New("failed to get memory info")
	}

	return info, nil
}

// InferenceManager functions

// NewInferenceManager initializes the inference manager
func NewInferenceManager(modelRepositoryPath string) (*InferenceManager, error) {
	cModelRepositoryPath := C.CString(modelRepositoryPath)
	defer C.free(unsafe.Pointer(cModelRepositoryPath))

	handle := C.InferenceInitialize(cModelRepositoryPath)
	if handle == nil {
		return nil, errors.New("failed to initialize inference manager")
	}

	manager := &InferenceManager{
		handle:       handle,
		loadedModels: make(map[string]*Model),
	}
	runtime.SetFinalizer(manager, (*InferenceManager).Shutdown)
	return manager, nil
}

// Shutdown shuts down the inference manager
func (im *InferenceManager) Shutdown() {
	im.loadedModelsMutex.Lock()
	defer im.loadedModelsMutex.Unlock()

	// Clean up any loaded models
	for _, model := range im.loadedModels {
		if model.handle != nil {
			C.ModelDestroy(model.handle)
		}
	}
	im.loadedModels = nil

	// Shutdown the inference manager
	if im.handle != nil {
		C.InferenceShutdown(im.handle)
		im.handle = nil
	}
}

// getModelKey creates a consistent key for the model map
func getModelKey(modelName, version string) string {
	if version == "" {
		return modelName
	}
	return fmt.Sprintf("%s:%s", modelName, version)
}

// LoadModel loads a model into the inference server
func (im *InferenceManager) LoadModel(modelName, version string) error {
	if im.handle == nil {
		return errors.New("inference manager not initialized")
	}

	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))

	var cVersion *C.char
	if version != "" {
		cVersion = C.CString(version)
		defer C.free(unsafe.Pointer(cVersion))
	}

	var cError C.ErrorMessage
	success := C.InferenceLoadModel(im.handle, cModelName, cVersion, &cError)
	if !success {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to load model")
		}
		return err
	}

	// After successful loading, create a Model instance and add to our map
	modelKey := getModelKey(modelName, version)

	// Create model config to use for creating the model
	config := ModelConfig{
		Name:    modelName,
		Version: version,
	}

	// Create the model
	model, err := createModelInternal(modelName, config, DeviceGPU, 0)
	if err != nil {
		return fmt.Errorf("model loaded in server but failed to create local handle: %v", err)
	}

	// Store it in our map
	im.loadedModelsMutex.Lock()
	im.loadedModels[modelKey] = model
	im.loadedModelsMutex.Unlock()

	return nil
}

// UnloadModel unloads a model from the inference server
func (im *InferenceManager) UnloadModel(modelName, version string) error {
	if im.handle == nil {
		return errors.New("inference manager not initialized")
	}

	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))

	var cVersion *C.char
	if version != "" {
		cVersion = C.CString(version)
		defer C.free(unsafe.Pointer(cVersion))
	}

	var cError C.ErrorMessage
	success := C.InferenceUnloadModel(im.handle, cModelName, cVersion, &cError)
	if !success {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to unload model")
		}
		return err
	}

	// Remove from our loaded models map
	modelKey := getModelKey(modelName, version)

	im.loadedModelsMutex.Lock()
	defer im.loadedModelsMutex.Unlock()

	if model, exists := im.loadedModels[modelKey]; exists {
		if model.handle != nil {
			C.ModelDestroy(model.handle)
		}
		delete(im.loadedModels, modelKey)
	}

	return nil
}

// IsModelLoaded checks if a model is loaded
func (im *InferenceManager) IsModelLoaded(modelName, version string) bool {
	if im.handle == nil {
		return false
	}

	cModelName := C.CString(modelName)
	defer C.free(unsafe.Pointer(cModelName))

	var cVersion *C.char
	if version != "" {
		cVersion = C.CString(version)
		defer C.free(unsafe.Pointer(cVersion))
	}

	return bool(C.InferenceIsModelLoaded(im.handle, cModelName, cVersion))
}

// ListModels lists all available models
func (im *InferenceManager) ListModels() []string {
	if im.handle == nil {
		return nil
	}

	var numModels C.int
	cModels := C.InferenceListModels(im.handle, &numModels)
	if cModels == nil || numModels == 0 {
		return nil
	}
	defer C.InferenceFreeModelList(cModels, numModels)

	models := make([]string, int(numModels))
	for i := 0; i < int(numModels); i++ {
		cModel := *(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cModels)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))
		models[i] = C.GoString(cModel)
	}

	return models
}

// GetModel gets a reference to an already loaded model
func (im *InferenceManager) GetModel(modelName, version string) (*Model, error) {
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Check if the model is loaded in the server
	if !im.IsModelLoaded(modelName, version) {
		return nil, fmt.Errorf("model '%s:%s' is not loaded", modelName, version)
	}

	// Lookup in our local map
	modelKey := getModelKey(modelName, version)

	im.loadedModelsMutex.RLock()
	model, exists := im.loadedModels[modelKey]
	im.loadedModelsMutex.RUnlock()

	if !exists {
		// If we don't have it in our map but it's loaded in the server,
		// create a new local model handle and add it to our map
		config := ModelConfig{
			Name:    modelName,
			Version: version,
		}

		var err error
		model, err = createModelInternal(modelName, config, DeviceGPU, 0)
		if err != nil {
			return nil, fmt.Errorf("failed to get handle for loaded model: %v", err)
		}

		// Add to our map
		im.loadedModelsMutex.Lock()
		im.loadedModels[modelKey] = model
		im.loadedModelsMutex.Unlock()
	}

	return model, nil
}

// RunInference executes inference using a loaded model
func (im *InferenceManager) RunInference(modelName string, version string, inputs []TensorData) ([]TensorData, error) {
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Get reference to the already loaded model
	model, err := im.GetModel(modelName, version)
	if err != nil {
		return nil, fmt.Errorf("failed to get model for inference: %v", err)
	}

	// Run inference using the existing model
	return model.Infer(inputs)
}

// Model functions

// createModelInternal creates a model handle for an already loaded model
func createModelInternal(modelPath string, config ModelConfig, deviceType DeviceType, deviceID int) (*Model, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	// Create input names array
	inputNames := make([]*C.char, len(config.InputNames))
	for i, name := range config.InputNames {
		inputNames[i] = C.CString(name)
		defer C.free(unsafe.Pointer(inputNames[i]))
	}

	// Create output names array
	outputNames := make([]*C.char, len(config.OutputNames))
	for i, name := range config.OutputNames {
		outputNames[i] = C.CString(name)
		defer C.free(unsafe.Pointer(outputNames[i]))
	}

	// Create C model config
	cConfig := C.ModelConfig{
		name:             C.CString(config.Name),
		version:          C.CString(config.Version),
		type_:            C.ModelType(config.Type),
		max_batch_size:   C.int(config.MaxBatchSize),
		input_names:      (**C.char)(unsafe.Pointer(&inputNames[0])),
		num_inputs:       C.int(len(config.InputNames)),
		output_names:     (**C.char)(unsafe.Pointer(&outputNames[0])),
		num_outputs:      C.int(len(config.OutputNames)),
		instance_count:   C.int(config.InstanceCount),
		dynamic_batching: C.bool(config.DynamicBatching),
	}
	defer C.free(unsafe.Pointer(cConfig.name))
	defer C.free(unsafe.Pointer(cConfig.version))

	var cError C.ErrorMessage
	handle := C.ModelCreate(cModelPath, C.ModelType(config.Type), &cConfig, C.DeviceType(deviceType), C.int(deviceID), &cError)
	if handle == nil {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to create model")
		}
		return nil, err
	}

	model := &Model{
		handle:  handle,
		name:    config.Name,
		version: config.Version,
	}
	return model, nil
}

// Destroy destroys a model instance
func (m *Model) Destroy() {
	if m.handle != nil {
		C.ModelDestroy(m.handle)
		m.handle = nil
	}
}

// Infer runs inference on the model
func (m *Model) Infer(inputs []TensorData) ([]TensorData, error) {
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	// Currently only supporting float32 for simplicity
	// In a full implementation, you would handle all data types

	// Create C input tensors
	cInputs := make([]C.TensorData, len(inputs))
	for i, input := range inputs {
		// Create shape
		shape := C.Shape{
			dims:     (*C.int64_t)(unsafe.Pointer(&input.Shape.Dims[0])),
			num_dims: C.int(len(input.Shape.Dims)),
		}

		// Handle data based on type
		var dataPtr unsafe.Pointer
		var dataSize C.size_t

		switch input.DataType {
		case DataTypeFloat32:
			if floatData, ok := input.Data.([]float32); ok {
				dataPtr = unsafe.Pointer(&floatData[0])
				dataSize = C.size_t(len(floatData) * 4) // 4 bytes per float32
			} else {
				return nil, errors.New("data type mismatch for input " + input.Name)
			}
		// Add cases for other supported data types

		default:
			return nil, errors.New("unsupported data type for input " + input.Name)
		}

		cInputs[i] = C.TensorData{
			name:      C.CString(input.Name),
			data_type: C.DataType(input.DataType),
			shape:     shape,
			data:      dataPtr,
			data_size: dataSize,
		}
		defer C.free(unsafe.Pointer(cInputs[i].name))
	}

	// Prepare output tensors (assuming we know the output shapes)
	// In practice, you might need to query the model for output shapes
	outputs := []TensorData{
		{
			Name:     "output",
			DataType: DataTypeFloat32,
			Shape:    Shape{Dims: []int64{1, 1000}}, // Example shape
			Data:     make([]float32, 1000),         // Pre-allocate output buffer
		},
	}

	cOutputs := make([]C.TensorData, len(outputs))
	for i, output := range outputs {
		// Create shape
		shape := C.Shape{
			dims:     (*C.int64_t)(unsafe.Pointer(&output.Shape.Dims[0])),
			num_dims: C.int(len(output.Shape.Dims)),
		}

		// Handle data based on type
		var dataPtr unsafe.Pointer
		var dataSize C.size_t

		switch output.DataType {
		case DataTypeFloat32:
			if floatData, ok := output.Data.([]float32); ok {
				dataPtr = unsafe.Pointer(&floatData[0])
				dataSize = C.size_t(len(floatData) * 4) // 4 bytes per float32
			} else {
				return nil, errors.New("data type mismatch for output " + output.Name)
			}
		// Add cases for other supported data types

		default:
			return nil, errors.New("unsupported data type for output " + output.Name)
		}

		cOutputs[i] = C.TensorData{
			name:      C.CString(output.Name),
			data_type: C.DataType(output.DataType),
			shape:     shape,
			data:      dataPtr,
			data_size: dataSize,
		}
		defer C.free(unsafe.Pointer(cOutputs[i].name))
	}

	// Run inference
	var cError C.ErrorMessage
	success := C.ModelInfer(
		m.handle,
		&cInputs[0], C.int(len(cInputs)),
		&cOutputs[0], C.int(len(cOutputs)),
		&cError,
	)

	if !success {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to run inference")
		}
		return nil, err
	}

	// Output data is already updated in the output slices since we passed pointers

	return outputs, nil
}

// GetMetadata gets metadata about the model
func (m *Model) GetMetadata() (*ModelMetadata, error) {
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	cMetadata := C.ModelGetMetadata(m.handle)
	if cMetadata == nil {
		return nil, errors.New("failed to get model metadata")
	}
	defer C.ModelFreeMetadata(cMetadata)

	// Convert inputs
	inputs := make([]string, int(cMetadata.num_inputs))
	for i := 0; i < int(cMetadata.num_inputs); i++ {
		cInput := *(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetadata.inputs)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))
		inputs[i] = C.GoString(cInput)
	}

	// Convert outputs
	outputs := make([]string, int(cMetadata.num_outputs))
	for i := 0; i < int(cMetadata.num_outputs); i++ {
		cOutput := *(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetadata.outputs)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))
		outputs[i] = C.GoString(cOutput)
	}

	metadata := &ModelMetadata{
		Name:        C.GoString(cMetadata.name),
		Version:     C.GoString(cMetadata.version),
		Type:        ModelType(cMetadata.model_type),
		Inputs:      inputs,
		Outputs:     outputs,
		Description: C.GoString(cMetadata.description),
		LoadTimeNs:  int64(cMetadata.load_time_ns),
	}

	return metadata, nil
}

// GetStats gets statistics about the model
func (m *Model) GetStats() (*ModelStats, error) {
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	cStats := C.ModelGetStats(m.handle)
	if cStats == nil {
		return nil, errors.New("failed to get model stats")
	}
	defer C.ModelFreeStats(cStats)

	stats := &ModelStats{
		InferenceCount:       int64(cStats.inference_count),
		TotalInferenceTimeNs: int64(cStats.total_inference_time_ns),
		LastInferenceTimeNs:  int64(cStats.last_inference_time_ns),
		MemoryUsageBytes:     uint64(cStats.memory_usage_bytes),
	}

	return stats, nil
}
