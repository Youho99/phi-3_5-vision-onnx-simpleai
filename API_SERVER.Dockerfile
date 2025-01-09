FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y python3-pip curl git git-lfs && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /app/
COPY serve_api.py models.toml .

RUN pip install -r /app/requirements.txt

WORKDIR .

EXPOSE 10999:10999

CMD ["python3", "serve_api.py"]
