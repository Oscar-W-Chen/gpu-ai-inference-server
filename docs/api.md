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

Returns information about all GPU devices.

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

Returns memory information for all GPU devices.

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

### List Models
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
      "name": "simple_model",
      "is_loaded": false,
      "state": "AVAILABLE"
    },
    {
      "name": "test_model",
      "is_loaded": true,
      "state": "AVAILABLE"
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
- `version`: Model version (query parameter, optional)

**Response**:
```json
{
  "name": "test_model",
  "version": "1",
  "is_loaded": true,
  "state": "LOADED",
  "repository_path": "./models/test_model",
  "inference_count": 42,
  "memory_usage_mb": 256,
  "last_inference_time_ms": 5.3
}
```

### Load Model
```
POST /models/{name}/load
```

Loads a model into memory.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (query parameter, optional)

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

Unloads a model from memory.

**Parameters**:
- `name`: Model name (path parameter)
- `version`: Model version (query parameter, optional)

**Response**:
```json
{
  "message": "Model unloaded successfully",
  "name": "test_model",
  "version": "1"
}
```

## Error Responses

All endpoints may return error responses with appropriate HTTP status codes:

- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Model or resource not found
- `500 Internal Server Error` - Server error

Error responses include a JSON object with an error message:

```json
{
  "error": "Model not found in repository"
}
```