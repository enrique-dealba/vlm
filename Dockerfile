# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.9
ENV VLLM_VERSION=0.5.1
ENV CUDA_VERSION=118

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN python${PYTHON_VERSION} -m pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install xFormers
RUN python${PYTHON_VERSION} -m pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Clone vLLM repository and install from source
RUN git clone https://github.com/vllm-project/vllm.git \
    && cd vllm \
    && git checkout v${VLLM_VERSION} \
    && python${PYTHON_VERSION} -m pip install -e .

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN python${PYTHON_VERSION} -m pip install -r requirements.txt

# Copy .env file and other files
COPY .env .env
COPY . .

# Make sure start.sh is executable
RUN chmod +x start.sh

# Expose port 8888
EXPOSE 8888

# Set start-up script as the entry point
ENTRYPOINT ["sh", "./start.sh"]
