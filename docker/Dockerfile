# syntax=docker/dockerfile:1.4
# Enable BuildKit features

# Build stage
FROM golang:1.23.6-alpine AS builder

# Add metadata for build stage
LABEL description="CPU-only Inference Server - Build Stage"

WORKDIR /app

# Add build dependencies
RUN apk add --no-cache curl

# Create a persistent module cache
ENV GOMODCACHE=/go/pkg/mod

# Copy go.mod and go.sum files first for better layer caching
COPY go.mod go.sum* ./

# Download dependencies (will be cached unless go.mod/go.sum changes)
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

# Copy source code
COPY . .

# Build with optimized flags
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -trimpath \
    -ldflags="-s -w" \
    -o /go/bin/server ./server/main.go

# Final stage - use a minimal Alpine image
FROM alpine:3.18

# Add metadata for final image
LABEL maintainer="oscar.william.chen@gmail.com"
LABEL version="1.0"
LABEL description="CPU-only Inference Server"

# Add runtime dependencies
RUN apk add --no-cache ca-certificates tzdata curl

# Create a non-root user to run the application
RUN adduser -D -u 10001 appuser
USER 10001

# Copy only the compiled binary from the builder stage
COPY --from=builder /go/bin/server /go/bin/server

EXPOSE 8080

# Keep the exact same CMD as before
CMD ["/go/bin/server"]