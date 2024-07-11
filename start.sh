#!/bin/sh

# Load configurations from .env file
export $(grep -v '^#' .env | xargs)

# Run FastAPI server for llm_server
python3.9 -m uvicorn vlm_server:app --host 0.0.0.0 --port 8888