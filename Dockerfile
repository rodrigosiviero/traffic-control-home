FROM openvino/ubuntu22_runtime:latest

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exportar modelo OpenVINO durante o build
RUN python -c "\
from ultralytics import YOLO; \
m = YOLO('yolov8n.pt'); \
m.export(format='openvino', imgsz=640, half=False)" \
    && ls -la yolov8n_openvino_model/

# Baixar modelo PT como fallback
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY src/ src/
COPY main.py .

RUN mkdir -p /data/clips /data/logs

ENV DISPLAY=""

EXPOSE 8090

CMD ["python", "main.py"]
