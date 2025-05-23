package binding

/*
#include <stdlib.h>
#include "inference_bridge.h"
#cgo CFLAGS: -I${SRCDIR}/../include
#cgo LDFLAGS: -L${SRCDIR}/../../build/inference_engine -linference_engine -lstdc++ -Wl,-rpath,${SRCDIR}/../../build/inference_engine
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

// MemoryInfo represents memory information for a GPU device
type MemoryInfo struct {
	Total uint64
	Free  uint64
	Used  uint64
}

// OutputConfig represents configuration for a model output
type OutputConfig struct {
	Name          string
	Shape         []int64
	Dims          []int64
	DataType      string
	LabelFilename string
}

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

// NewInferenceManager initializes the inference manager
//
// This creates a new inference manager that will manage models in the provided
// model repository path. The manager handles loading, unloading, and inference
// for models.
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

// Shutdown shuts down the inference manager and releases all resources
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
//
// Loads the specified model and version into memory. If version is empty,
// the latest version will be loaded. After successful loading, the model
// will be available for inference.
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

	// Get a reference to the already loaded model instead of creating a new one
	im.loadedModelsMutex.Lock()
	defer im.loadedModelsMutex.Unlock()

	// Create a model reference that points to the already loaded model
	model := &Model{
		name:    modelName,
		version: version,
	}

	// Get a handle to the already-loaded model
	var getHandleError C.ErrorMessage
	modelHandle := C.GetModelHandle(im.handle, cModelName, cVersion, &getHandleError)
	if modelHandle == nil {
		var err error
		if getHandleError != nil {
			err = errors.New(C.GoString(getHandleError))
			C.FreeErrorMessage(getHandleError)
		} else {
			err = errors.New("failed to get model handle")
		}
		return fmt.Errorf("model loaded in server but failed to get handle: %v", err)
	}

	model.handle = modelHandle
	im.loadedModels[modelKey] = model

	return nil
}

// UnloadModel unloads a model from the inference server
//
// Unloads the specified model and version from memory. If version is empty,
// the latest loaded version will be unloaded. After unloading, the model
// will no longer be available for inference.
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
//
// Returns true if the specified model and version is currently loaded and
// ready for inference. If version is empty, checks if any version of the
// model is loaded.
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
//
// Returns a list of all model names available in the model repository,
// regardless of whether they are currently loaded or not.
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
//
// Returns a Model object representing the specified model name and version,
// if it is already loaded. If version is empty, gets the latest loaded version.
// Returns an error if the model is not loaded.
func (im *InferenceManager) GetModel(modelName, version string) (*Model, error) {
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Check if the model is loaded in the server
	isLoaded := im.IsModelLoaded(modelName, version)
	if !isLoaded {
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
//
// Runs inference on the specified model and version using the provided input tensors.
// The output configurations specify the expected outputs. Returns the resulting
// output tensors or an error if inference fails.
func (im *InferenceManager) RunInference(modelName string, version string, inputs []TensorData, outputConfigs []OutputConfig) ([]TensorData, error) {
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Get reference to the already loaded model
	model, err := im.GetModel(modelName, version)
	if err != nil {
		return nil, fmt.Errorf("failed to get model for inference: %v", err)
	}

	// Run inference using the existing model, passing output configurations
	return model.Infer(inputs, outputConfigs)
}

// createModelInternal creates a model handle for an already loaded model
func createModelInternal(modelPath string, config ModelConfig, deviceType DeviceType, deviceID int) (*Model, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	cName := C.CString(config.Name)
	defer C.free(unsafe.Pointer(cName))

	cVersion := C.CString(config.Version)
	defer C.free(unsafe.Pointer(cVersion))

	// Create C model config
	cConfig := C.ModelConfig{
		name:             cName,
		version:          cVersion,
		type_:            C.ModelType(config.Type),
		max_batch_size:   C.int(config.MaxBatchSize),
		instance_count:   C.int(config.InstanceCount),
		dynamic_batching: C.bool(config.DynamicBatching),
	}

	// Handle input names array safely
	var inputNamesPtr **C.char
	if len(config.InputNames) > 0 {
		inputNames := make([]*C.char, len(config.InputNames))
		for i, name := range config.InputNames {
			inputNames[i] = C.CString(name)
			defer C.free(unsafe.Pointer(inputNames[i]))
		}
		inputNamesPtr = (**C.char)(unsafe.Pointer(&inputNames[0]))
	}
	cConfig.input_names = inputNamesPtr
	cConfig.num_inputs = C.int(len(config.InputNames))

	// Handle output names array safely
	var outputNamesPtr **C.char
	if len(config.OutputNames) > 0 {
		outputNames := make([]*C.char, len(config.OutputNames))
		for i, name := range config.OutputNames {
			outputNames[i] = C.CString(name)
			defer C.free(unsafe.Pointer(outputNames[i]))
		}
		outputNamesPtr = (**C.char)(unsafe.Pointer(&outputNames[0]))
	}
	cConfig.output_names = outputNamesPtr
	cConfig.num_outputs = C.int(len(config.OutputNames))

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

// Infer runs inference on the model
//
// Executes inference using this model with the provided input tensors.
// The output configurations specify the expected outputs. Returns the
// resulting output tensors or an error if inference fails.
func (m *Model) Infer(inputs []TensorData, outputConfigs []OutputConfig) ([]TensorData, error) {
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	// Check if model is loaded by directly calling C function
	isLoaded := bool(C.ModelIsLoaded(m.handle))
	if !isLoaded {
		return nil, errors.New("model not loaded")
	}

	if len(inputs) == 0 {
		return nil, errors.New("no input tensors provided")
	}

	// Create output tensor objects based on outputConfigs
	outputs := make([]TensorData, len(outputConfigs))
	for i, outConfig := range outputConfigs {
		var shape []int64
		if len(outConfig.Shape) > 0 {
			shape = outConfig.Shape
		} else if len(outConfig.Dims) > 0 {
			shape = outConfig.Dims
		} else {
			return nil, fmt.Errorf("no shape defined for output '%s'", outConfig.Name)
		}

		var outputData interface{}
		var dataType DataType

		switch outConfig.DataType {
		case "FLOAT32", "TYPE_FP32":
			dataType = DataTypeFloat32
			// Calculate total elements needed
			elements := int64(1)
			for _, dim := range shape {
				elements *= dim
			}
			outputData = make([]float32, elements)
		default:
			return nil, fmt.Errorf("unsupported data type '%s' for output '%s'", outConfig.DataType, outConfig.Name)
		}

		outputs[i] = TensorData{
			Name:     outConfig.Name,
			DataType: dataType,
			Shape:    Shape{Dims: shape},
			Data:     outputData,
		}
	}

	// Determine counts
	inputCount := len(inputs)
	outputCount := len(outputs)

	// Allocate arrays for C names (for inputs and outputs)
	inputNames := make([]*C.char, inputCount)
	outputNames := make([]*C.char, outputCount)

	// Ensure C strings are freed after use
	defer func() {
		for i := 0; i < inputCount; i++ {
			if inputNames[i] != nil {
				C.free(unsafe.Pointer(inputNames[i]))
			}
		}
		for i := 0; i < outputCount; i++ {
			if outputNames[i] != nil {
				C.free(unsafe.Pointer(outputNames[i]))
			}
		}
	}()

	// Allocate arrays for C tensor data structures
	cInputs := make([]C.TensorData, inputCount)
	cOutputs := make([]C.TensorData, outputCount)

	// Track C memory allocations (for shapes, data buffers, etc.) to free later.
	var cMemoryToFree []unsafe.Pointer
	defer func() {
		for _, ptr := range cMemoryToFree {
			C.free(ptr)
		}
	}()

	// Process input tensors
	for i, input := range inputs {
		// Convert the input name to a C string
		inputNames[i] = C.CString(input.Name)
		cInputs[i].name = inputNames[i]
		cInputs[i].data_type = C.DataType(input.DataType)

		// Allocate C memory for the shape dims array
		if len(input.Shape.Dims) > 0 {
			dimsCount := len(input.Shape.Dims)
			dimsSize := C.size_t(dimsCount) * C.size_t(unsafe.Sizeof(C.int64_t(0)))
			cDims := C.malloc(dimsSize)
			if cDims == nil {
				return nil, errors.New("failed to allocate memory for input shape dims")
			}
			// Create a slice backed by the C memory and copy dims into it.
			dimsSlice := (*[1 << 30]C.int64_t)(cDims)[:dimsCount:dimsCount]
			for j, dim := range input.Shape.Dims {
				dimsSlice[j] = C.int64_t(dim)
			}
			cInputs[i].shape.dims = (*C.int64_t)(cDims)
			cInputs[i].shape.num_dims = C.int(dimsCount)
			cMemoryToFree = append(cMemoryToFree, cDims)
		}

		// Process input data (currently handling float32 only)
		switch input.DataType {
		case DataTypeFloat32:
			floatData, ok := input.Data.([]float32)
			if !ok || len(floatData) == 0 {
				return nil, errors.New("invalid float32 data for input " + input.Name)
			}
			dataSize := len(floatData) * int(unsafe.Sizeof(float32(0)))
			dataPtr := C.malloc(C.size_t(dataSize))
			if dataPtr == nil {
				return nil, errors.New("failed to allocate memory for input data")
			}
			cMemoryToFree = append(cMemoryToFree, dataPtr)
			cFloatArray := (*[1 << 30]float32)(dataPtr)[:len(floatData):len(floatData)]
			copy(cFloatArray, floatData)
			cInputs[i].data = dataPtr
			cInputs[i].data_size = C.size_t(dataSize)
		default:
			return nil, errors.New("unsupported data type for input " + input.Name)
		}
	}

	// Process output tensors
	for i, output := range outputs {
		// Convert the output name to a C string
		outputNames[i] = C.CString(output.Name)
		cOutputs[i].name = outputNames[i]
		cOutputs[i].data_type = C.DataType(output.DataType)

		// Allocate C memory for the shape dims array
		if len(output.Shape.Dims) > 0 {
			dimsCount := len(output.Shape.Dims)
			dimsSize := C.size_t(dimsCount) * C.size_t(unsafe.Sizeof(C.int64_t(0)))
			cDims := C.malloc(dimsSize)
			if cDims == nil {
				return nil, errors.New("failed to allocate memory for output shape dims")
			}
			dimsSlice := (*[1 << 30]C.int64_t)(cDims)[:dimsCount:dimsCount]
			for j, dim := range output.Shape.Dims {
				dimsSlice[j] = C.int64_t(dim)
			}
			cOutputs[i].shape.dims = (*C.int64_t)(cDims)
			cOutputs[i].shape.num_dims = C.int(dimsCount)
			cMemoryToFree = append(cMemoryToFree, cDims)
		}

		// Allocate memory for output data (float32 only)
		switch output.DataType {
		case DataTypeFloat32:
			floatData, ok := output.Data.([]float32)
			if !ok || len(floatData) == 0 {
				return nil, errors.New("invalid float32 data for output " + output.Name)
			}
			dataSize := len(floatData) * int(unsafe.Sizeof(float32(0)))
			dataPtr := C.malloc(C.size_t(dataSize))
			if dataPtr == nil {
				return nil, errors.New("failed to allocate memory for output data")
			}
			cMemoryToFree = append(cMemoryToFree, dataPtr)
			cOutputs[i].data = dataPtr
			cOutputs[i].data_size = C.size_t(dataSize)
		default:
			return nil, errors.New("unsupported data type for output " + output.Name)
		}
	}

	// Call the C API to run inference
	var cError C.ErrorMessage
	success := C.ModelInfer(
		m.handle,
		&cInputs[0], C.int(inputCount),
		&cOutputs[0], C.int(outputCount),
		&cError,
	)
	if !bool(success) {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to run inference")
		}
		return nil, err
	}

	// Copy the output data from C memory back into the Go slices
	for i, output := range outputs {
		switch output.DataType {
		case DataTypeFloat32:
			floatData, ok := output.Data.([]float32)
			if !ok {
				return nil, errors.New("invalid output data for " + output.Name)
			}
			dataSize := int(cOutputs[i].data_size)
			numElements := dataSize / int(unsafe.Sizeof(float32(0)))
			cFloatArray := (*[1 << 30]float32)(cOutputs[i].data)[:numElements:numElements]
			copy(floatData, cFloatArray)
		default:
			return nil, errors.New("unsupported data type for output " + output.Name)
		}
	}

	return outputs, nil
}

// GetMetadata gets metadata about the model
//
// Returns model metadata including name, version, input/output names, and more.
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
//
// Returns statistics about the model, including inference counts and timing.
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
