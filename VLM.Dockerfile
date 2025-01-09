FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Install Python, pip, curl, et git
RUN apt-get update && \
    apt-get install -y python3-pip curl git git-lfs && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Installer les paquets Python nécessaires
RUN pip install onnxruntime-genai-cuda

RUN pip install huggingface-hub transformers

# Copy requirements.txt and the local package directory (simpleAI)
COPY app/requirements.txt /app/
COPY app/simpleAI /app/simpleAI
COPY app/get_models.py /app/get_models.py

# Install Python dependencies from requirements.txt
RUN pip install -r /app/requirements.txt
RUN python3 app/get_models.py

# Install local package in editable mode
RUN pip install -e /app/simpleAI

# Set working directory
WORKDIR /app

COPY app/ .

# Publish port
EXPOSE 50051:50051

# Enjoy
ENTRYPOINT python3 server.py
CMD ["--address", "[::]:50051"]
