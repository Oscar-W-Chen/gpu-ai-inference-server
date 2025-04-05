import requests
import json
import time
import argparse
import numpy as np
import base64
from PIL import Image
import io
import sys

# Default to localhost, but allow override via command line argument
parser = argparse.ArgumentParser(description='Test GPU Inference Server API')
parser.add_argument('--url', default='http://localhost:8080', help='Base URL of the inference server')
parser.add_argument('--test-inference', action='store_true', help='Run inference test with sample data')
parser.add_argument('--image', default=None, help='Path to image file for inference test')
parser.add_argument('--model', default=None, help='Specific model to test')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
args = parser.parse_args()

BASE_URL = args.url
VERBOSE = args.verbose

def debug_print(*messages):
    """Print only if verbose mode is enabled"""
    if VERBOSE:
        print(*messages)

def test_health():
    print("\n=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_cuda_info():
    print("\n=== Testing CUDA Info Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/cuda")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_devices():
    print("\n=== Testing Devices Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/devices")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_memory_info():
    print("\n=== Testing GPU Memory Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/gpu/memory")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_models():
    print("\n=== Testing Models List Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/models")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        
        # Extract model names for other tests
        if 'models' in response.json():
            return [model['name'] for model in response.json()['models']]
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_model_config(model_name, version=None):
    """
    Get detailed model configuration from the server
    Returns a dictionary with the model's configuration or None if not found
    """
    try:
        url = f"{BASE_URL}/models/{model_name}"
        if version:
            url += f"?version={version}"
            
        response = requests.get(url)
        if response.status_code != 200:
            debug_print(f"Failed to get model info: {response.status_code}")
            debug_print(response.text)
            return None
        
        model_info = response.json()
        debug_print(f"Model config details: {json.dumps(model_info, indent=2)}")
        return model_info
    except Exception as e:
        debug_print(f"Error getting model config: {e}")
        return None

def test_model_status(model_name):
    print(f"\n=== Testing Model Status for '{model_name}' ===")
    try:
        response = requests.get(f"{BASE_URL}/models/{model_name}")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_load_model(model_name):
    print(f"\n=== Testing Model Load for '{model_name}' ===")
    try:
        response = requests.post(f"{BASE_URL}/models/{model_name}/load")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_unload_model(model_name):
    print(f"\n=== Testing Model Unload for '{model_name}' ===")
    try:
        response = requests.post(f"{BASE_URL}/models/{model_name}/unload")
        print("Status:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_lifecycle(model_name):
    print(f"\n=== Testing Full Model Lifecycle for '{model_name}' ===")
    
    # Get initial status
    test_model_status(model_name)
    
    # Load model
    test_load_model(model_name)
    
    # Check status after loading
    test_model_status(model_name)
    
    # Try loading again (should say already loaded)
    test_load_model(model_name)
    
    # Unload model
    test_unload_model(model_name)
    
    # Check status after unloading
    test_model_status(model_name)
    
    # Try unloading again (should say not loaded)
    test_unload_model(model_name)

def prepare_image_data(image_path, input_shape):
    """
    Prepare image data for inference based on the required input shape.
    Resizes and processes the image to fit the model's requirements.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Default to standard size if shape is incomplete
        width, height = 224, 224
        
        # Extract dimensions from shape if available
        if len(input_shape) >= 4:  # Batch, channels, height, width
            height, width = input_shape[2], input_shape[3]
        elif len(input_shape) >= 2:  # Height, width
            height, width = input_shape[0], input_shape[1]
        
        print(f"Resizing image to {width}x{height} to match model requirements")
        img = img.resize((width, height))
        img_array = np.array(img).astype(np.float32)
        
        # Normalize image (simple normalization for testing)
        img_array = img_array / 255.0
        
        # Reshape based on expected input format
        if len(input_shape) >= 4 and input_shape[1] == 3:
            # Likely expects [batch, channels, height, width] - NCHW format
            img_array = img_array.transpose(2, 0, 1)  # CHW format
            
        # Ensure batch dimension if needed
        if len(input_shape) >= 4 and input_shape[0] == 1:
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Flatten for JSON serialization
        flat_data = img_array.flatten().tolist()
        
        return flat_data
    except Exception as e:
        print(f"Error preparing image: {e}")
        return None

def generate_dynamic_dummy_data(input_config):
    """
    Generate appropriate dummy data based on model input configuration
    """
    try:
        if not input_config:
            print("No input configuration available, using default data")
            return np.random.rand(10).astype(np.float32).tolist()
        
        # Extract shape information
        input_shape = input_config.get('shape', input_config.get('dims', []))
        if not input_shape:
            print("No shape information in config, using default data")
            return np.random.rand(10).astype(np.float32).tolist()
        
        # Generate data of appropriate shape
        shape = input_shape
        print(f"Generating dummy data with shape: {shape}")
        
        # Create dummy data with the right shape and type
        data_type = input_config.get('data_type', 'FLOAT32')
        if data_type == 'FLOAT32' or data_type == 'FP32':
            dummy_data = np.random.rand(*shape).astype(np.float32)
        elif data_type == 'INT32':
            dummy_data = np.random.randint(0, 100, size=shape).astype(np.int32)
        elif data_type == 'INT64':
            dummy_data = np.random.randint(0, 100, size=shape).astype(np.int64)
        else:
            print(f"Unsupported data type: {data_type}, using float32")
            dummy_data = np.random.rand(*shape).astype(np.float32)
            
        # Flatten for JSON serialization
        return dummy_data.flatten().tolist()
    except Exception as e:
        print(f"Error generating dummy data: {e}")
        print(f"Input config was: {input_config}")
        return np.random.rand(10).astype(np.float32).tolist()  # Fallback

def select_test_model(model_names, preferred_model=None):
    """
    Select the most appropriate model to test based on availability and preference
    """
    if not model_names:
        return None
        
    # If user specified a model, try to use it
    if preferred_model:
        for model in model_names:
            if model == preferred_model or model.startswith(f"{preferred_model}/"):
                return model
        print(f"Warning: Specified model '{preferred_model}' not found")
    
    # Default to first model if no preference or preference not found
    return model_names[0]

def is_image_model(model_config):
    """
    Determine if a model is likely an image classification model
    based on its input configuration
    """
    if not model_config or 'config' not in model_config:
        return False
        
    if 'inputs' not in model_config['config']:
        return False
        
    for input_config in model_config['config']['inputs']:
        # Look for input shape that might represent image data
        shape = input_config.get('shape', input_config.get('dims', []))
        
        # Check for NCHW format (common for image models)
        if len(shape) >= 3 and shape[-3] == 3:  # 3 channels
            return True
            
    return False

def test_inference(model_name, image_path=None):
    """
    Test model inference using either provided image or dynamically generated data
    """
    print(f"\n=== Testing Inference for '{model_name}' ===")
    
    # First make sure model is loaded
    response = requests.post(f"{BASE_URL}/models/{model_name}/load")
    if response.status_code not in [200, 202]:
        print(f"Failed to load model: {response.status_code}")
        print(response.text)
        return False
    
    # Get model config to understand model requirements
    model_config = get_model_config(model_name)
    if not model_config:
        print(f"Failed to get model configuration, cannot perform inference")
        return False
    
    # Check if model config has been returned by the server
    if 'config' not in model_config:
        print("Warning: Model config details not available in server response")
        print("The server might need to be updated to return config information")
        return False
        
    # Extract input configuration
    if 'inputs' not in model_config['config'] or not model_config['config']['inputs']:
        print("Error: No input configuration found for the model")
        return False
        
    # Get the first input configuration (we'll just test with the first input for simplicity)
    input_config = model_config['config']['inputs'][0]
    input_name = input_config.get('name', 'input')
    input_shape = input_config.get('shape', input_config.get('dims', []))
    
    print(f"Using input '{input_name}' with shape {input_shape}")
    
    # Determine if we should use image processing
    should_use_image = image_path and is_image_model(model_config)
    
    # Prepare input data
    if should_use_image:
        print(f"Processing image for model: {model_name}")
        input_data = prepare_image_data(image_path, input_shape)
        if input_data is None:
            return False
    else:
        if image_path and not is_image_model(model_config):
            print(f"Note: Model {model_name} doesn't appear to be an image model")
            print(f"Using generated dummy data instead of processing the image")
            
        input_data = generate_dynamic_dummy_data(input_config)
    
    # Craft inference request
    request_data = {
        "inputs": {
            input_name: input_data
        }
    }
    
    # Send inference request
    try:
        response = requests.post(
            f"{BASE_URL}/models/{model_name}/infer",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        print("Inference Status:", response.status_code)
        if response.status_code == 200:
            result = response.json()
            print("Inference Result (truncated):")
            
            # If classifications are available, show them
            for output in result.get('outputs', []):
                if 'classifications' in output:
                    print("\nTop predictions:")
                    for i, pred in enumerate(output['classifications']):
                        if 'label' in pred:
                            print(f"  {i+1}. {pred['label']} ({pred['probability']:.4f})")
                        else:
                            print(f"  {i+1}. Class {pred['index']} ({pred['probability']:.4f})")
                else:
                    # For other outputs, show summary
                    data = output.get('data', [])
                    if isinstance(data, list) and len(data) > 10:
                        print(f"Output '{output['name']}': {len(data)} values, "
                              f"first 5: {data[:5]}, "
                              f"shape: {output.get('shape', 'unknown')}")
                    else:
                        print(f"Output '{output['name']}': {data}")
            
            return True
        else:
            print("Error response:", response.text)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print(f"Testing against server at: {BASE_URL}")
    
    # Test basic server endpoints
    test_health()
    test_cuda_info()
    
    # Test GPU-related endpoints
    try:
        test_devices()
        test_memory_info()
    except Exception as e:
        print(f"GPU endpoint tests failed (expected if no GPU): {e}")
    
    # Test model management endpoints
    model_names = test_models()
    
    if model_names:
        # Select a model to test
        test_model = select_test_model(model_names, args.model)
        
        if test_model:
            # Test the model's lifecycle
            test_model_lifecycle(test_model)
            
            # Optionally run inference test
            if args.test_inference:
                if args.image:
                    print(f"Testing inference on {test_model} with image: {args.image}")
                    test_inference(test_model, args.image)
                else:
                    print(f"Testing inference on {test_model} with dynamically generated data")
                    test_inference(test_model)
        else:
            print("No suitable model found for testing")
    else:
        print("\nNo models found in repository. Cannot test model management endpoints.")
        print("Please make sure you have models in the ./models directory.")