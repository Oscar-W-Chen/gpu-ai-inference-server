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

// Model functions

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

// Destroy destroys a model instance
func (m *Model) Destroy() {
	if m.handle != nil {
		C.ModelDestroy(m.handle)
		m.handle = nil
	}
}

// This function replaces the Model.Infer method in inference_binding.go
func (m *Model) Infer(inputs []TensorData, outputConfigs []OutputConfig) ([]TensorData, error) {
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	// First, check that we have inputs
	if len(inputs) == 0 {
		return nil, errors.New("no input tensors provided")
	}

	// Debug input information
	for i, input := range inputs {
		fmt.Printf("Input tensor %d: name=%s, shape=%v, dataType=%d\n",
			i, input.Name, input.Shape.Dims, input.DataType)
	}

	// Debug output configuration
	fmt.Println("Output configurations:")
	for i, outConfig := range outputConfigs {
		fmt.Printf("Output %d: name=%s, shape=%v, dataType=%s\n",
			i, outConfig.Name, outConfig.Shape, outConfig.DataType)
	}

	// Create output tensors based on provided output configurations
	outputs := make([]TensorData, len(outputConfigs))
	for i, outConfig := range outputConfigs {
		// Get shape from config
		var shape []int64
		if len(outConfig.Shape) > 0 {
			shape = outConfig.Shape
		} else if len(outConfig.Dims) > 0 {
			shape = outConfig.Dims
		} else {
			return nil, fmt.Errorf("no shape defined for output '%s'", outConfig.Name)
		}

		// Create buffer for output data based on data type
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
			fmt.Printf("Created output buffer for '%s' with %d elements\n", outConfig.Name, elements)
		// Add cases for other supported data types
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

	// The key issue is the cgo pointer restriction. We need to copy all data
	// to C-allocated memory and then copy results back to Go memory.

	// First, create C input and output tensor arrays for batch processing
	inputCount := len(inputs)
	outputCount := len(outputs)

	// Allocate C memory for string names that will be passed to C
	inputNames := make([]*C.char, inputCount)
	outputNames := make([]*C.char, outputCount)

	// Memory to be freed later
	defer func() {
		for i := 0; i < inputCount; i++ {
			C.free(unsafe.Pointer(inputNames[i]))
		}
		for i := 0; i < outputCount; i++ {
			C.free(unsafe.Pointer(outputNames[i]))
		}
	}()

	// Create arrays for holding input and output shapes in C memory
	var inputShapeArrays [][]C.int64_t
	var outputShapeArrays [][]C.int64_t

	// Allocate memory for C tensors
	cInputs := make([]C.TensorData, inputCount)
	cOutputs := make([]C.TensorData, outputCount)

	// Track C memory allocations for cleanup
	var cMemoryToFree []unsafe.Pointer
	defer func() {
		for _, ptr := range cMemoryToFree {
			C.free(ptr)
		}
	}()

	// Process input tensors to C format
	for i, input := range inputs {
		// Convert names to C strings
		inputNames[i] = C.CString(input.Name)
		cInputs[i].name = inputNames[i]
		cInputs[i].data_type = C.DataType(input.DataType)

		// Process shapes
		if len(input.Shape.Dims) > 0 {
			// Allocate shape array in C memory
			shapeArray := make([]C.int64_t, len(input.Shape.Dims))
			for j, dim := range input.Shape.Dims {
				shapeArray[j] = C.int64_t(dim)
			}
			inputShapeArrays = append(inputShapeArrays, shapeArray)

			// Set shape in the C struct
			cInputs[i].shape.dims = &inputShapeArrays[len(inputShapeArrays)-1][0]
			cInputs[i].shape.num_dims = C.int(len(input.Shape.Dims))
		}

		// Process data based on type
		switch input.DataType {
		case DataTypeFloat32:
			if floatData, ok := input.Data.([]float32); ok && len(floatData) > 0 {
				// Calculate data size
				dataSize := len(floatData) * int(unsafe.Sizeof(float32(0)))

				// Allocate C memory for the data
				dataPtr := C.malloc(C.size_t(dataSize))
				cMemoryToFree = append(cMemoryToFree, dataPtr)

				// Copy Go data to C memory
				// This is safer than using Go memory directly
				cFloatArray := (*[1 << 30]float32)(dataPtr)[:len(floatData):len(floatData)]
				copy(cFloatArray, floatData)

				// Set data pointer in C struct
				cInputs[i].data = dataPtr
				cInputs[i].data_size = C.size_t(dataSize)
			} else {
				return nil, errors.New("invalid float32 data for input " + input.Name)
			}
		default:
			return nil, errors.New("unsupported data type for input " + input.Name)
		}
	}

	// Process output tensors similarly
	for i, output := range outputs {
		// Convert names to C strings
		outputNames[i] = C.CString(output.Name)
		cOutputs[i].name = outputNames[i]
		cOutputs[i].data_type = C.DataType(output.DataType)

		// Process shapes
		if len(output.Shape.Dims) > 0 {
			// Allocate shape array in C memory
			shapeArray := make([]C.int64_t, len(output.Shape.Dims))
			for j, dim := range output.Shape.Dims {
				shapeArray[j] = C.int64_t(dim)
			}
			outputShapeArrays = append(outputShapeArrays, shapeArray)

			// Set shape in the C struct
			cOutputs[i].shape.dims = &outputShapeArrays[len(outputShapeArrays)-1][0]
			cOutputs[i].shape.num_dims = C.int(len(output.Shape.Dims))
		}

		// Allocate memory for output data
		switch output.DataType {
		case DataTypeFloat32:
			// Get the output slice
			floatData, ok := output.Data.([]float32)
			if !ok || len(floatData) == 0 {
				return nil, errors.New("invalid float32 data for output " + output.Name)
			}

			// Calculate data size
			dataSize := len(floatData) * int(unsafe.Sizeof(float32(0)))

			// Allocate C memory for the output data
			dataPtr := C.malloc(C.size_t(dataSize))
			cMemoryToFree = append(cMemoryToFree, dataPtr)

			// Set data pointer in C struct
			cOutputs[i].data = dataPtr
			cOutputs[i].data_size = C.size_t(dataSize)
		default:
			return nil, errors.New("unsupported data type for output " + output.Name)
		}
	}

	// Execute inference using C API
	var cError C.ErrorMessage
	success := C.ModelInfer(
		m.handle,
		&cInputs[0], C.int(inputCount),
		&cOutputs[0], C.int(outputCount),
		&cError,
	)

	// Handle error
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

	// Copy output data back to Go memory
	for i, output := range outputs {
		switch output.DataType {
		case DataTypeFloat32:
			floatData, ok := output.Data.([]float32)
			if !ok {
				return nil, errors.New("invalid output buffer")
			}

			// Copy C memory back to Go slice
			cFloatArray := (*[1 << 30]float32)(cOutputs[i].data)[:len(floatData):len(floatData)]
			copy(floatData, cFloatArray)
		}
	}

	return outputs, nil
}

// Note: This function replaces the previous runModelInfer, so you can remove that function

// GetMetadata gets metadata about the model
func (m *Model) GetMetadata() (*ModelMetadata, error) {
	if m.handle == nil {
		fmt.Println("DEBUG: GetMetadata - model handle is nil")
		return nil, errors.New("model not initialized")
	}

	fmt.Println("DEBUG: GetMetadata - calling C.ModelGetMetadata")
	cMetadata := C.ModelGetMetadata(m.handle)
	if cMetadata == nil {
		fmt.Println("DEBUG: GetMetadata - C.ModelGetMetadata returned nil")
		return nil, errors.New("failed to get model metadata")
	}
	fmt.Printf("DEBUG: GetMetadata - C.ModelGetMetadata returned successfully: num_inputs=%d, num_outputs=%d\n",
		cMetadata.num_inputs, cMetadata.num_outputs)
	defer C.ModelFreeMetadata(cMetadata)

	// Convert inputs
	inputs := make([]string, int(cMetadata.num_inputs))
	fmt.Printf("DEBUG: GetMetadata - Processing %d inputs\n", int(cMetadata.num_inputs))
	for i := 0; i < int(cMetadata.num_inputs); i++ {
		cInput := *(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetadata.inputs)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))
		inputs[i] = C.GoString(cInput)
		fmt.Printf("DEBUG: GetMetadata - Input %d: %s\n", i, inputs[i])
	}

	// Convert outputs
	outputs := make([]string, int(cMetadata.num_outputs))
	fmt.Printf("DEBUG: GetMetadata - Processing %d outputs\n", int(cMetadata.num_outputs))
	for i := 0; i < int(cMetadata.num_outputs); i++ {
		cOutput := *(**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cMetadata.outputs)) + uintptr(i)*unsafe.Sizeof(uintptr(0))))
		outputs[i] = C.GoString(cOutput)
		fmt.Printf("DEBUG: GetMetadata - Output %d: %s\n", i, outputs[i])
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

	fmt.Printf("DEBUG: GetMetadata - Returning metadata: Name=%s, Version=%s, Type=%d, Inputs=%v, Outputs=%v\n",
		metadata.Name, metadata.Version, metadata.Type, metadata.Inputs, metadata.Outputs)

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
