if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU is available"
    nvidia-smi
else
    echo "GPU is not available"
    exit 1
fi