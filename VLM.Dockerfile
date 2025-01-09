FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip curl git git-lfs && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip install onnxruntime-genai-cuda

RUN pip install huggingface-hub transformers

COPY app/requirements.txt /app/
COPY app/get_models.py /app/get_models.py

RUN pip install -r /app/requirements.txt
RUN python3 app/get_models.py

WORKDIR /app

COPY app/ .

EXPOSE 50051:50051

ENTRYPOINT python3 server.py
CMD ["--address", "[::]:50051"]
