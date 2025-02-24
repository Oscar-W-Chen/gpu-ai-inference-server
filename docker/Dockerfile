FROM ubuntu:22.04

# Add metadata
LABEL maintainer="oscar.william.chen@gmail.com"
LABEL version="1.0"
LABEL description="CPU-only Inference Server"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV GOPATH="/go"
ENV PATH="/usr/local/go/bin:${GOPATH}/bin:${PATH}"

# Configure apt
RUN echo "Acquire::http::Pipeline-Depth 0;" > /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::http::No-Cache true;" >> /etc/apt/apt.conf.d/99custom && \
    echo "Acquire::BrokenProxy    true;" >> /etc/apt/apt.conf.d/99custom

# Install all required packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip==24.2

# Install specific Go version (1.23.6)
RUN wget -O go.tgz https://go.dev/dl/go1.23.6.linux-amd64.tar.gz && \
    rm -rf /usr/local/go && \
    tar -C /usr/local -xzf go.tgz && \
    rm go.tgz

WORKDIR /app

# Copy only necessary files
COPY go.mod go.sum ./
COPY server ./server/

# Download dependencies and build
RUN go mod download
RUN CGO_ENABLED=0 GOOS=linux go build -o /app/server-binary ./server/main.go && \
    chmod +x /app/server-binary

# Expose server port
EXPOSE 8080

# Run the server using the full path
CMD ["/app/server-binary"]