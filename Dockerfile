FROM python:3.11-slim

# Dependências do sistema + Intel GPU (VA-API / OpenCL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    intel-opencl-icd \
    intel-media-va-driver \
    i965-va-driver \
    clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements com OpenVINO
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exportar modelo OpenVINO durante o build
# Isso gera yolov8n_openvino_model/ com best.xml + best.bin
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
# OpenVINO usa /dev/dri pra Intel GPU
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

EXPOSE 8090

CMD ["python", "main.py"]
