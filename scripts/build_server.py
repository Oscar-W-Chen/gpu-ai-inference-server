#!/usr/bin/env python3
"""
Simple test script for Google Colab to test the C++ inference engine
Assumes the repository is already cloned/uploaded to Colab
"""

import os
import time
import subprocess
import requests
import json
from IPython.display import display, HTML

def run_cmd(cmd, show_output=True):
    """Run a shell command and print output"""
    print(f"Running: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Stream output in real-time
    all_output = []
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output:
            line = output.strip()
            all_output.append(line)
            if show_output:
                print(line)
    
    # Get any remaining output
    remaining_output, errors = proc.communicate()
    if remaining_output:
        all_output.extend(remaining_output.strip().split('\n'))
        if show_output:
            print(remaining_output.strip())
    
    if errors and show_output:
        print(f"Errors: {errors}")
    
    return '\n'.join(all_output), proc.returncode

def check_cuda():
    """Check if CUDA is available in the Colab environment"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA device count: {device_count}")
            for i in range(device_count):
                print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
        return cuda_available
    except ImportError:
        print("PyTorch not available. Installing...")
        run_cmd("pip install torch")
        import torch
        return torch.cuda.is_available()

def setup_environment():
    """Setup the test environment"""
    print("\n=== Setting up environment ===")
    
    # Install Go if needed
    go_version, rc = run_cmd("go version", show_output=False)
    if rc != 0:
        print("Installing Go...")
        os.system("wget https://dl.google.com/go/go1.23.6.linux-amd64.tar.gz")
        run_cmd("tar -C /usr/local -xzf go1.23.6.linux-amd64.tar.gz")
        run_cmd("rm go1.23.6.linux-amd64.tar.gz")
        os.environ['PATH'] += ':/usr/local/go/bin'
        
        
        # Verify Go installation
        go_version, rc = run_cmd("go version")
        if rc != 0:
            print("Failed to install Go")
            return False
    else:
        print(f"Go is already installed: {go_version}")
    
    # Install required packages for building
    print("Installing required packages...")
    run_cmd("apt-get update && apt-get install -y build-essential cmake git pkg-config")
    
    # Create build directories
    print("Creating build directories...")
    run_cmd("mkdir -p build/inference_engine")
    
    return True

def build_inference_engine():
    """Build the C++ inference engine"""
    print("\n=== Building inference engine ===")
    
    # Make build script executable
    run_cmd("chmod +x scripts/build_inference_engine.sh")
    
    # Run build script
    run_cmd("./scripts/build_inference_engine.sh")
    
    # Verify the lib was created
    if os.path.exists("build/inference_engine/lib/libinference_engine.so"):
        print("✅ Inference engine built successfully")
        return True
    else:
        print("❌ Failed to build inference engine")
        return False

def build_go_server():
    """Build the Go server"""
    print("\n=== Building Go server ===")
    
    # Set Go environment variables to ensure the library can be found
    current_dir = os.getcwd()
    lib_path = f"{current_dir}/build/inference_engine/lib"
    
    os.environ["CGO_ENABLED"] = "1"
    os.environ["CGO_LDFLAGS"] = f"-L{lib_path} -linference_engine"
    
    print(f"Using library path: {lib_path}")
    print(f"CGO_LDFLAGS set to: {os.environ['CGO_LDFLAGS']}")
    
    # Verify the library exists
    if not os.path.exists(f"{lib_path}/libinference_engine.so"):
        print(f"Warning: Cannot find {lib_path}/libinference_engine.so")
        run_cmd(f"ls -la {lib_path}")
    
    # Build the server
    output, rc = run_cmd("go build -o gpu-ai-inference-server ./server/main.go")
    
    if rc == 0 and os.path.exists("gpu-ai-inference-server"):
        print("✅ Go server built successfully")
        return True
    else:
        print("❌ Failed to build Go server")
        return False
'''
def run_server():
    """Run the server and test it"""
    print("\n=== Running server ===")
    
    # Set LD_LIBRARY_PATH to find the shared library
    os.environ["LD_LIBRARY_PATH"] = f"{os.getcwd()}/build/inference_engine/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Start the server
    server_process = subprocess.Popen("./server", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print("Server started")
    
    # Wait for the server to start
    time.sleep(2)

    # Test the server
    try:
        print("\n=== Testing server API ===")

        # Check if server is actually running
        run_cmd("curl -v http://localhost:8080/health --ipv4")
        
        # Test health endpoint
        response = requests.get("http://localhost:8080/health")
        print(f"Health endpoint response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        # Test CUDA info endpoint
        response = requests.get("http://localhost:8080/cuda")
        print(f"CUDA info endpoint response: {response.status_code}")
        cuda_info = response.json()
        print(json.dumps(cuda_info, indent=2))
        
        if cuda_info.get("cuda_available", False):
            # Test devices endpoint
            response = requests.get("http://localhost:8080/devices")
            print(f"Devices endpoint response: {response.status_code}")
            print(json.dumps(response.json(), indent=2))
    
    except Exception as e:
        print(f"Error testing server: {e}")
    
    finally:
        # Stop the server
        server_process.terminate()
        server_process.wait()
        print("Server stopped")
    
'''
def main():
    """Main function"""
    print("=== GPU AI Inference Server Test ===")
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Setup environment
    if setup_environment():
        # Build inference engine
        if build_inference_engine():
            # Build Go server
            build_go_server();
        else:
            print("Skipping server build and test due to inference engine build failure")
    else:
        print("Environment setup failed")

if __name__ == "__main__":
    main()