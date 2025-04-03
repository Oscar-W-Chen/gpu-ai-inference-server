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
	"log"
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
	log.Printf("DEBUG: LoadModel called for model %s version %s", modelName, version)
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
	log.Printf("DEBUG: Calling C.InferenceLoadModel for %s:%s", modelName, version)
	success := C.InferenceLoadModel(im.handle, cModelName, cVersion, &cError)
	log.Printf("DEBUG: C.InferenceLoadModel returned: success=%v", success)
	if !success {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to load model")
		}
		log.Printf("DEBUG: LoadModel failed with error: %v", err)
		return err
	}

	// After successful loading, create a Model instance and add to our map
	modelKey := getModelKey(modelName, version)
	log.Printf("DEBUG: Model successfully loaded, creating model key: %s", modelKey)

	// Create model config to use for creating the model
	config := ModelConfig{
		Name:    modelName,
		Version: version,
	}

	// Create the model
	log.Printf("DEBUG: Creating model handle via createModelInternal")
	model, err := createModelInternal(modelName, config, DeviceGPU, 0)
	if err != nil {
		log.Printf("DEBUG: Failed to create model handle: %v", err)
		return fmt.Errorf("model loaded in server but failed to create local handle: %v", err)
	}
	log.Printf("DEBUG: Model handle created successfully: %p", model.handle)

	// Store it in our map
	im.loadedModelsMutex.Lock()
	im.loadedModels[modelKey] = model
	log.Printf("DEBUG: Model added to loadedModels map with key: %s", modelKey)
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
	log.Printf("DEBUG: GetModel called for model %s version %s", modelName, version)
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Check if the model is loaded in the server
	isLoaded := im.IsModelLoaded(modelName, version)
	log.Printf("DEBUG: IsModelLoaded returned: %v", isLoaded)
	if !isLoaded {
		return nil, fmt.Errorf("model '%s:%s' is not loaded", modelName, version)
	}

	// Lookup in our local map
	modelKey := getModelKey(modelName, version)
	log.Printf("DEBUG: Looking up model with key: %s", modelKey)

	im.loadedModelsMutex.RLock()
	model, exists := im.loadedModels[modelKey]
	im.loadedModelsMutex.RUnlock()

	log.Printf("DEBUG: Model exists in map: %v, model handle: %p", exists, model)
	if !exists {
		// If we don't have it in our map but it's loaded in the server,
		// create a new local model handle and add it to our map
		log.Printf("DEBUG: Model not found in map, creating new handle")
		config := ModelConfig{
			Name:    modelName,
			Version: version,
		}

		var err error
		model, err = createModelInternal(modelName, config, DeviceGPU, 0)
		if err != nil {
			log.Printf("DEBUG: Failed to create new model handle: %v", err)
			return nil, fmt.Errorf("failed to get handle for loaded model: %v", err)
		}
		log.Printf("DEBUG: Created new model handle: %p", model.handle)

		// Add to our map
		im.loadedModelsMutex.Lock()
		im.loadedModels[modelKey] = model
		log.Printf("DEBUG: Added new model handle to map with key: %s", modelKey)
		im.loadedModelsMutex.Unlock()
	}

	return model, nil
}

// RunInference executes inference using a loaded model
func (im *InferenceManager) RunInference(modelName string, version string, inputs []TensorData, outputConfigs []OutputConfig) ([]TensorData, error) {
	log.Printf("DEBUG: RunInference called for model %s version %s", modelName, version)
	if im.handle == nil {
		return nil, errors.New("inference manager not initialized")
	}

	// Get reference to the already loaded model
	log.Printf("DEBUG: Getting model for inference")
	model, err := im.GetModel(modelName, version)
	if err != nil {
		log.Printf("DEBUG: Failed to get model for inference: %v", err)
		return nil, fmt.Errorf("failed to get model for inference: %v", err)
	}
	log.Printf("DEBUG: Got model handle: %p", model.handle)

	// Run inference using the existing model, passing output configurations
	log.Printf("DEBUG: Calling model.Infer with %d inputs and %d output configs", len(inputs), len(outputConfigs))
	return model.Infer(inputs, outputConfigs)
}

// Model functions

// createModelInternal creates a model handle for an already loaded model
func createModelInternal(modelPath string, config ModelConfig, deviceType DeviceType, deviceID int) (*Model, error) {
	log.Printf("DEBUG: createModelInternal called for path %s", modelPath)
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
	log.Printf("DEBUG: Calling C.ModelCreate")
	handle := C.ModelCreate(cModelPath, C.ModelType(config.Type), &cConfig, C.DeviceType(deviceType), C.int(deviceID), &cError)
	log.Printf("DEBUG: C.ModelCreate returned handle: %p", handle)
	if handle == nil {
		var err error
		if cError != nil {
			err = errors.New(C.GoString(cError))
			C.FreeErrorMessage(cError)
		} else {
			err = errors.New("failed to create model")
		}
		log.Printf("DEBUG: Model creation failed: %v", err)
		return nil, err
	}

	// Check if the model is loaded
	log.Printf("DEBUG: Checking if newly created model is loaded")
	isLoaded := bool(C.ModelIsLoaded(handle))
	log.Printf("DEBUG: Newly created model loaded state: %v", isLoaded)

	// If model is not loaded, try loading it
	if !isLoaded {
		log.Printf("DEBUG: Model not loaded, attempting to load via C.ModelLoad")
		success := C.ModelLoad(handle, &cError)
		if !success {
			var err error
			if cError != nil {
				err = errors.New(C.GoString(cError))
				C.FreeErrorMessage(cError)
			} else {
				err = errors.New("failed to load model")
			}
			log.Printf("DEBUG: Model loading failed: %v", err)
			return nil, err
		}
		log.Printf("DEBUG: Model loaded successfully")
	}

	model := &Model{
		handle:  handle,
		name:    config.Name,
		version: config.Version,
	}
	log.Printf("DEBUG: Created new Model instance with handle: %p", model.handle)
	return model, nil
}

// Destroy destroys a model instance
func (m *Model) Destroy() {
	if m.handle != nil {
		C.ModelDestroy(m.handle)
		m.handle = nil
	}
}

// Runs Inference on the model
func (m *Model) Infer(inputs []TensorData, outputConfigs []OutputConfig) ([]TensorData, error) {
	log.Printf("DEBUG: Model.Infer called on model handle: %p", m.handle)
	if m.handle == nil {
		return nil, errors.New("model not initialized")
	}

	// Check if model is loaded by directly calling C function
	log.Printf("DEBUG: Checking if model is loaded via C.ModelIsLoaded")
	isLoaded := bool(C.ModelIsLoaded(m.handle))
	log.Printf("DEBUG: C.ModelIsLoaded returned: %v", isLoaded)
	if !isLoaded {
		return nil, errors.New("model not loaded")
	}

	if len(inputs) == 0 {
		return nil, errors.New("no input tensors provided")
	}

	// Debug logging for inputs
	for i, input := range inputs {
		fmt.Printf("Input tensor %d: name=%s, shape=%v, dataType=%d\n",
			i, input.Name, input.Shape.Dims, input.DataType)
	}

	// Debug logging for outputs
	fmt.Println("Output configurations:")
	for i, outConfig := range outputConfigs {
		fmt.Printf("Output %d: name=%s, shape=%v, dataType=%s\n",
			i, outConfig.Name, outConfig.Shape, outConfig.DataType)
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
			fmt.Printf("Created output buffer for '%s' with %d elements\n", outConfig.Name, elements)
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
