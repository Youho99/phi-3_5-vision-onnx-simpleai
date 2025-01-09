FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Install Python, pip, curl, et git
RUN apt-get update && \
    apt-get install -y python3-pip curl git git-lfs && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and the local package directory (simpleAI)
COPY app/simpleAI /simpleAI
COPY serve_api.py models.toml .

# Install local package in editable mode
RUN pip install grpcio==1.64.1 grpcio-tools==1.64.1 
# torch
RUN pip install -e /simpleAI

# Set working directory
WORKDIR .

# Publish port
EXPOSE 10999:10999

# Command to run the application
CMD ["python3", "serve_api.py"]
