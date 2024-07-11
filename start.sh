#!/bin/bash

# Load configurations from .env file
export $(grep -v '^#' .env | xargs)

# Check for GPU availability
if ! command -v nvidia-smi &> /dev/null || [ -z "$(nvidia-smi -L)" ]; then
    echo "Error: No GPU available. VLLM requires GPU acceleration."
    exit 1
fi

# Run FastAPI server for llm_server
python3.9 -m uvicorn vlm_server:app --host 0.0.0.0 --port 8888