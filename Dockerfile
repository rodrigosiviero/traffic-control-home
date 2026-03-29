FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Docker usa headless (sem GUI)
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Baixar modelo YOLO durante build
RUN python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); m.export(format='onnx', imgsz=640, simplify=True, opset=12)" \
    && mkdir -p /root/.cache/traffic-monitor \
    && mv yolov8n.onnx /root/.cache/traffic-monitor/yolov8n.onnx

COPY src/ src/
COPY main.py .

RUN mkdir -p /data/clips /data/logs

ENV DISPLAY=""

EXPOSE 8090

CMD ["python", "main.py"]
