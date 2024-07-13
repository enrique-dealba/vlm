#!/bin/bash

# Load configurations from .env file
export $(grep -v '^#' .env | xargs)

# Check for GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Ensure NVIDIA drivers are installed."
    exit 1
fi

if [ -z "$(nvidia-smi -L)" ]; then
    echo "Error: No GPU detected by nvidia-smi."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
if [ -z "$CUDA_VERSION" ]; then
    echo "Error: CUDA not found or nvcc not in PATH."
    exit 1
fi

echo "CUDA Version: $CUDA_VERSION"

# Run FastAPI server for llm_server
python3.10 -m uvicorn vlm_server:app --host 0.0.0.0 --port 8888