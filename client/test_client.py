import requests
import json
import time
import argparse

# Default to localhost, but allow override via command line argument
parser = argparse.ArgumentParser(description='Test GPU Inference Server API')
parser.add_argument('--url', default='http://localhost:8080', help='Base URL of the inference server')
args = parser.parse_args()

BASE_URL = args.url

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
        # Test the first model's lifecycle
        test_model_lifecycle(model_names[0])
    else:
        print("\nNo models found in repository. Cannot test model management endpoints.")
        print("Please make sure you have models in the ./models directory.")