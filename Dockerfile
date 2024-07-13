# Use NVIDIA CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV VLLM_VERSION=0.5.1
ENV CUDA_VERSION=118

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN python${PYTHON_VERSION} -m ensurepip \
    && python${PYTHON_VERSION} -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM with CUDA 11.8
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp${PYTHON_VERSION/./}-cp${PYTHON_VERSION/./}-manylinux1_x86_64.whl

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir -r requirements.txt

# Copy .env file and other files
COPY .env .env
COPY . .

# Make sure start.sh is executable
RUN chmod +x start.sh

# Expose port 8888
EXPOSE 8888

# Set start-up script as the entry point
ENTRYPOINT ["sh", "./start.sh"]
