# Multi-stage build for development environment
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 as dev

# Add metadata
LABEL maintainer="oscar.william.chen@gmail.com"
LABEL version="1.0"
LABEL description="Development environment with CUDA support"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    golang \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip==24.2

# Install Go and development tools in a single layer
RUN wget https://go.dev/dl/go1.22.0.linux-amd64.tar.gz \
    && rm -rf /usr/local/go \
    && tar -C /usr/local -xzf go1.22.0.linux-amd64.tar.gz \
    && rm go1.22.0.linux-amd64.tar.gz 

# Set up environment variables
ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV PATH="${GOPATH}/bin:${PATH}"

# Install common development tools
RUN go install github.com/go-delve/delve/cmd/dlv@latest
RUN go install golang.org/x/tools/gopls@latest

# Create user and set up directories
RUN useradd -m developer && \
    mkdir -p /workspace /go && \
    chown -R developer:developer /workspace /go

# Set up working directory
WORKDIR /workspace

# Add GPU check script
COPY scripts/gpu_check.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/gpu_check.sh

# Health check for GPU availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nvidia-smi || exit 1

# Switch to non-root user
USER developer

# Default command
CMD ["/bin/bash"]