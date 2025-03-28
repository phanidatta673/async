# Use an ARM64-compatible base image
FROM --platform=linux/arm64 python:3.10-bullseye

# Set environment variables for better memory management
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and activate a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install PyTorch separately from its correct index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies from PyPI
RUN pip install --no-cache-dir fastapi uvicorn pydantic huggingface_hub

# Install ctransformers from source for ARM64 compatibility
RUN pip install --no-cache-dir git+https://github.com/marella/ctransformers.git

# Create app directory
WORKDIR /app

# Copy only the necessary files
COPY api/docker_vllm_server.py ./api/
COPY scripts/test_api.py ./scripts/
COPY data ./data

# Expose port
EXPOSE 8001

# Command to run the server
CMD ["python3", "-X", "utf8", "api/docker_vllm_server.py"]
