# Cell 1: Test the health endpoint
import requests
import json

response = requests.get("http://localhost:8080/health")
print(f"Health endpoint response: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# Cell 2: Test CUDA info endpoint
response = requests.get("http://localhost:8080/cuda")
print(f"CUDA info endpoint response: {response.status_code}")
cuda_info = response.json()
print(json.dumps(cuda_info, indent=2))

# Cell 3: Test devices endpoint if CUDA is available
if cuda_info.get("cuda_available", False):
    response = requests.get("http://localhost:8080/devices")
    print(f"Devices endpoint response: {response.status_code}")
    print(json.dumps(response.json(), indent=2))