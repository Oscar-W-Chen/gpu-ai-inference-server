#!/bin/bash

echo -e "\n=== Running server ==="

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$(pwd)/build/inference_engine/lib:${LD_LIBRARY_PATH}"

# Run the GPU AI Inference Server with NGROK_AUTHTOKEN
NGROK_AUTHTOKEN="2uP6AQR3CAdUtbDDpw89QbPv2xg_5RZzMYmiXn4KFGXJgQ5ZX" ./gpu-ai-inference-server