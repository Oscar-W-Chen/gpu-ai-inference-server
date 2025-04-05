# GPU AI Inference Server API Documentation

This document describes the REST API endpoints provided by the GPU AI Inference Server.

## Base Endpoints

### Health Check
```
GET /health
```
Returns the health status of the server.

**Response**:
```json
{
  "status": "healthy",
  "time": 1679569872
}
```

### CUDA Information
```
GET /cuda
```
Returns information about CUDA availability and device count.

**Response**:
```json
{
  "cuda_available": true,
  "device_count": 1
}
```

## Device Information

### List Devices
```
GET /devices
```
Returns information about all available GPU devices.

**Response**:
```json
{
  "devices": [
    "Device 0: Tesla T4 (Compute Capability 7.5)"
  ]
}
```

### GPU Memory Information
```
GET /gpu/memory
```
Returns detailed memory usage for GPU devices.

**Response**:
```json
{
  "memory_info": [
    {
      "device_id": 0,
      "total_mb": 15360,
      "free_mb": 14336,
      "used_mb": 1024,
      "used_percent": 6.66
    }
  ]
}
```

## Model Management

### List Available Models
```
GET /models
```
Returns a list of all models available in the repository.

**Response**:
```json
{
  "repository_path": "./models",
  "model_count": 2,
  "models": [
    {
      "name": "test_model",
      "is_loaded": false,
      "state": "AVAILABLE"
    },
    {
      "name": "densenet_onnx",
      "is_loaded": true,
      "state": "LOADED"
    }
  ]
}
```

### Get Model Status
```
GET /models/{name}
```
Returns detailed status information for a specific model.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (optional query parameter)

**Response**:
```json
{
  "name": "test_model",
  "version": "1",
  "is_loaded": true,
  "state": "LOADED",
  "repository_path": "./models/test_model",
  "available_versions": ["1"],
  "config": {
    "name": "test_model",
    "version": "1",
    "inputs": [
      {
        "name": "input",
        "shape": [1, 3],
        "data_type": "FLOAT32"
      }
    ],
    "outputs": [
      {
        "name": "output",
        "shape": [1, 2],
        "data_type": "FLOAT32"
      }
    ]
  }
}
```

### Load Model
```
POST /models/{name}/load
```
Loads a specific model into memory.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (optional query parameter)

**Response**:
```json
{
  "message": "Model loaded successfully",
  "name": "test_model",
  "version": "1"
}
```

### Unload Model
```
POST /models/{name}/unload
```
Unloads a specific model from memory.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (optional query parameter)

**Response**:
```json
{
  "message": "Model unloaded successfully",
  "name": "test_model",
  "version": "1"
}
```

### Run Inference
```
POST /models/{name}/infer
```
Run inference on a loaded model.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (optional query parameter)

**Request Body**:
```json
{
  "inputs": {
    "input": [1.0, 2.0, 3.0]
  }
}
```

**Response**:
```json
{
  "model_name": "test_model",
  "model_version": "1",
  "outputs": [
    {
      "name": "output",
      "data_type": "FLOAT32",
      "shape": [1, 2],
      "data": [4.5, 5.5]
    }
  ]
}
```

## Error Handling

All endpoints may return error responses with appropriate HTTP status codes:

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Model or resource not found
- `500 Internal Server Error`: Server-side error

**Error Response Format**:
```json
{
  "error": "Detailed error message"
}
```

## Notes

- Model names are case-sensitive
- If no version is specified, the latest version of the model will be used
- Ensure models are loaded before running inference
- Supported model types: ONNX (currently), with planned support for TensorFlow, TensorRT, and PyTorch

## Authentication

Currently, this API does not require authentication. Future versions may implement security features.