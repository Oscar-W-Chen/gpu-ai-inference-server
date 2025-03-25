#!/bin/bash
# Script to download and install ONNX Runtime

# Exit on error
set -e

# ONNX Runtime version to install
ONNXRUNTIME_VERSION="1.16.3"

# Detect OS and architecture
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

# Map architecture to ONNX Runtime naming
if [ "$ARCH" == "x86_64" ]; then
    ARCH="x64"
elif [ "$ARCH" == "aarch64" ]; then
    ARCH="arm64"
fi

# Set installation directory
INSTALL_DIR="/usr/local/onnxruntime"

# Create temporary directory
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "Downloading ONNX Runtime v${ONNXRUNTIME_VERSION} for ${OS}-${ARCH}..."

# Construct download URL
if [ "$OS" == "linux" ]; then
    if [ "$ARCH" == "x64" ]; then
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz"
    elif [ "$ARCH" == "arm64" ]; then
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
elif [ "$OS" == "darwin" ]; then
    if [ "$ARCH" == "x64" ]; then
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-x64-${ONNXRUNTIME_VERSION}.tgz"
    elif [ "$ARCH" == "arm64" ]; then
        URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz"
    else
        echo "Unsupported architecture: $ARCH"
        exit 1
    fi
else
    echo "Unsupported operating system: $OS"
    exit 1
fi

# Download and extract
echo "Downloading from $URL"
curl -L $URL -o onnxruntime.tgz
tar -xzf onnxruntime.tgz

# Get the extracted directory name
EXTRACTED_DIR=$(ls | grep "onnxruntime")

# Create installation directory
sudo mkdir -p $INSTALL_DIR
sudo mkdir -p $INSTALL_DIR/include
sudo mkdir -p $INSTALL_DIR/lib

# Copy header files
echo "Installing ONNX Runtime headers..."
sudo cp -r $EXTRACTED_DIR/include/* $INSTALL_DIR/include/

# Copy library files
echo "Installing ONNX Runtime libraries..."
sudo cp -r $EXTRACTED_DIR/lib/* $INSTALL_DIR/lib/

# Set permissions
sudo chmod -R 755 $INSTALL_DIR

# Clean up
cd ..
rm -rf $TMP_DIR

echo "ONNX Runtime v${ONNXRUNTIME_VERSION} installed successfully to $INSTALL_DIR"
echo "To use it with CMake, set -DONNXRUNTIME_ROOT=$INSTALL_DIR"