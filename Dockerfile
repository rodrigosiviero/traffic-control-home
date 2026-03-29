FROM python:3.11-slim

# Dependências básicas do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    gnupg \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU compute repo (OpenCL / Level Zero para iGPU)
RUN curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key \
    | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
    https://repositories.intel.com/gpu/ubuntu jammy client" \
    > /etc/apt/sources.list.d/intel-graphics.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements com OpenVINO
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
