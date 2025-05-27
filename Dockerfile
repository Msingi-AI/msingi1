FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN python3.10 -m pip install --upgrade pip

# Copy project files
COPY . .

# Install the package
RUN pip install .

# Set environment variables
ENV PYTHONPATH=/workspace/src

# Default command
CMD ["python", "-m", "msingi1.train_with_shards"]
