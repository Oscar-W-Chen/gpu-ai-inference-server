#!/bin/bash
# Simple script to install ONNX Runtime v1.21.0 (GPU)

set -e

INSTALL_DIR="/usr/local/onnxruntime"
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "Downloading ONNX Runtime v1.21.0..."
curl -L -o onnxruntime.tgz "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz"

tar -xzf onnxruntime.tgz
EXTRACTED_DIR=$(ls -d */ | grep "onnxruntime")

sudo mkdir -p "$INSTALL_DIR"/{include,lib}
sudo cp -r "$EXTRACTED_DIR/include/"* "$INSTALL_DIR/include/"
sudo cp -r "$EXTRACTED_DIR/lib/"* "$INSTALL_DIR/lib/"
sudo chmod -R 755 "$INSTALL_DIR"

cd ..
rm -rf "$TMP_DIR"

echo "ONNX Runtime v1.21.0 installed successfully to $INSTALL_DIR"
echo "To use it with CMake, set -DONNXRUNTIME_ROOT=$INSTALL_DIR"