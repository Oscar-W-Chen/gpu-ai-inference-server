import requests
import numpy as np 
import json
import time

BASE_URL = "https://c107-34-125-208-73.ngrok-free.app"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health check:", response.json())

def test_cuda_info():
    response = requests.get(f"{BASE_URL}/cuda")
    print("CUDA info:", response.json())

def test_devices():
    response = requests.get(f"{BASE_URL}/devices")
    print("Devices:", response.json())

def test_memory_info():
    response = requests.get(f"{BASE_URL}/gpu/memory")
    print("Memory info:", response.json())

def test_models():
    response = requests.get(f"{BASE_URL}/models")
    print("Models:", response.json())

if __name__ == "__main__":
    test_health()
    test_cuda_info()
    test_devices()
    test_memory_info()
    test_models()