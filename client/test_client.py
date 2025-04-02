#!/usr/bin/env python3
import argparse
import json
import requests
import time
import sys

# Parse command line arguments
parser = argparse.ArgumentParser(description='Simple test client for GPU AI Inference Server')
parser.add_argument('--url', required=True, help='Base URL of the inference server')
parser.add_argument('--model', required=True, help='Model name to test')
parser.add_argument('--version', default='', help='Model version (optional)')
parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
args = parser.parse_args()

BASE_URL = args.url
MODEL_NAME = args.model
VERSION = args.version
VERBOSE = args.verbose

def log(message):
    """Print message if verbose mode is enabled"""
    if VERBOSE:
        print(f"DEBUG: {message}")

# Step 1: Check server health
print(f"Testing server health at {BASE_URL}...")
try:
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print(f"✅ Server is healthy: {response.json()}")
    else:
        print(f"❌ Server health check failed: {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"❌ Could not connect to server: {e}")
    sys.exit(1)

# Step 2: Check if model exists
print(f"Checking if model '{MODEL_NAME}' exists...")
try:
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code != 200:
        print(f"❌ Failed to list models: {response.status_code}")
        sys.exit(1)
        
    models_data = response.json()
    log(f"Models response: {json.dumps(models_data, indent=2)}")
    
    # Check if model exists in the list
    model_exists = False
    if 'models' in models_data:
        for model in models_data['models']:
            if model['name'] == MODEL_NAME:
                model_exists = True
                print(f"✅ Model '{MODEL_NAME}' found")
                log(f"Model details: {model}")
                break
                
    if not model_exists:
        print(f"❌ Model '{MODEL_NAME}' not found")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Failed to check model existence: {e}")
    sys.exit(1)

# Step 3: Ensure model is loaded
print(f"Ensuring model '{MODEL_NAME}' is loaded...")
try:
    # Check model status first
    status_url = f"{BASE_URL}/models/{MODEL_NAME}"
    if VERSION:
        status_url += f"?version={VERSION}"
        
    response = requests.get(status_url)
    if response.status_code != 200:
        print(f"❌ Failed to get model status: {response.status_code}")
        log(f"Response: {response.text}")
        sys.exit(1)
        
    model_status = response.json()
    log(f"Model status: {json.dumps(model_status, indent=2)}")
    
    # If model is not loaded, try to load it
    if not model_status.get('is_loaded', False):
        print(f"Model '{MODEL_NAME}' is not loaded. Attempting to load...")
        
        load_url = f"{BASE_URL}/models/{MODEL_NAME}/load"
        if VERSION:
            load_url += f"?version={VERSION}"
            
        response = requests.post(load_url)
        if response.status_code not in [200, 202]:
            print(f"❌ Failed to load model: {response.status_code}")
            log(f"Response: {response.text}")
            sys.exit(1)
            
        print(f"✅ Model loading initiated: {response.json()}")
        
        # Wait for model to load
        print("Waiting for model to become available...")
        for _ in range(10):  # Try up to 10 times
            time.sleep(2)  # Wait 2 seconds between checks
            response = requests.get(status_url)
            if response.status_code == 200:
                model_status = response.json()
                if model_status.get('is_loaded', False):
                    print("✅ Model is now loaded")
                    break
            print(".", end="", flush=True)
        else:
            print("\n❌ Timeout waiting for model to load")
            sys.exit(1)
    else:
        print("✅ Model is already loaded")
        
except Exception as e:
    print(f"❌ Failed to ensure model is loaded: {e}")
    sys.exit(1)

# Step 4: Extract model config to prepare inference data
print("Extracting model input configuration...")
try:
    if 'config' not in model_status:
        print("❌ Model configuration not available in status response")
        sys.exit(1)
        
    model_config = model_status['config']
    inputs_config = model_config.get('inputs', [])
    if not inputs_config:
        print("❌ No input configuration found for model")
        sys.exit(1)
        
    # Get the first input for simplicity
    input_config = inputs_config[0]
    input_name = input_config.get('name', 'input')
    input_shape = input_config.get('shape', input_config.get('dims', []))
    input_type = input_config.get('data_type', 'FLOAT32')
    
    print(f"Using input '{input_name}' with shape {input_shape}, type {input_type}")
    
    # Calculate total elements needed
    total_elements = 1
    for dim in input_shape:
        total_elements *= dim
        
    # Generate simple test data (all ones or sequential values)
    if input_type in ['FLOAT32', 'FP32']:
        # Create sequential values
        input_data = [float(i) / total_elements for i in range(total_elements)]
    else:
        # Default to zeros for other types
        input_data = [0] * total_elements
        
    log(f"Generated {len(input_data)} elements for input")
    
except Exception as e:
    print(f"❌ Failed to extract model configuration: {e}")
    sys.exit(1)

# Step 5: Prepare and send inference request
print("Sending inference request...")
try:
    # Prepare request payload
    request_data = {
        "inputs": {
            input_name: input_data
        }
    }
    
    log(f"Request payload: {json.dumps(request_data)}")
    
    # Send the request
    infer_url = f"{BASE_URL}/models/{MODEL_NAME}/infer"
    if VERSION:
        infer_url += f"?version={VERSION}"
        
    response = requests.post(
        infer_url,
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    # Process response
    if response.status_code == 200:
        result = response.json()
        print("✅ Inference successful!")
        
        # Print outputs
        print("\nResults:")
        for output in result.get('outputs', []):
            print(f"Output: {output['name']}")
            
            # If classifications are available, show them
            if 'classifications' in output:
                print("  Top predictions:")
                for i, pred in enumerate(output['classifications']):
                    if 'label' in pred:
                        print(f"    {i+1}. {pred['label']} ({pred['probability']:.4f})")
                    else:
                        print(f"    {i+1}. Class {pred['index']} ({pred['probability']:.4f})")
            else:
                # For other outputs, show summary
                data = output.get('data', [])
                if isinstance(data, list):
                    if len(data) > 10:
                        print(f"  Data: {len(data)} values, first 5: {data[:5]}, shape: {output.get('shape', 'unknown')}")
                    else:
                        print(f"  Data: {data}")
                        
    else:
        print(f"❌ Inference failed with status code {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Failed to send inference request: {e}")
    sys.exit(1)

print("\nTest completed.")