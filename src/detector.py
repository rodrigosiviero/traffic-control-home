import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("traffic-monitor.detector")

# Mapeamento COCO class IDs → nomes
COCO_VEHICLES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Detecção de veículos usando YOLO (otimizado para CPU)."""
    
    def __init__(self, config: dict):
        self.model_size = config.get("model_size", "n")
        self.confidence = config.get("confidence", 0.4)
        self.target_classes = config.get("classes", [2, 3, 5, 7])
        self.use_onnx = config.get("use_onnx", True)
        
        self.class_names = {k: COCO_VEHICLES[k] for k in self.target_classes if k in COCO_VEHICLES}
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo YOLO otimizado para CPU."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("Ultralytics não instalado. pip install ultralytics")
            raise
        
        if self.use_onnx:
            # ONNX é mais rápido em CPU
            model_name = f"yolov8{self.model_size}.onnx"
            model_path = Path.home() / ".cache" / "traffic-monitor" / model_name
            
            if not model_path.exists():
                logger.info(f"Exportando modelo para ONNX: {model_name}")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                temp_model = YOLO(f"yolov8{self.model_size}.pt")
                temp_model.export(format="onnx", imgsz=640, simplify=True, opset=12)
                # Mover para cache
                import shutil
                exported = Path(f"yolov8{self.model_size}.onnx")
                if exported.exists():
                    shutil.move(str(exported), str(model_path))
            
            logger.info(f"Carregando modelo ONNX: {model_path}")
            self.model = YOLO(str(model_path))
        else:
            model_name = f"yolov8{self.model_size}.pt"
            logger.info(f"Carregando modelo PyTorch: {model_name}")
            self.model = YOLO(model_name)
        
        # Warmup - primeira inferência é sempre mais lenta
        logger.info("Aquecendo modelo (warmup)...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            conf=self.confidence,
            classes=self.target_classes,
            verbose=False,
            device="cpu",
        )
        logger.info("Modelo pronto!")
    
    def detect(self, frame: np.ndarray) -> list:
        """
        Detecta veículos no frame.
        
        Returns:
            Lista de detecções: [[x1, y1, x2, y2, confidence, class_id], ...]
        """
        results = self.model.predict(
            frame,
            conf=self.confidence,
            classes=self.target_classes,
            verbose=False,
            device="cpu",
            half=False,
            max_det=20,  # Limitar detecções pra economizar CPU
        )
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    detections.append([x1, y1, x2, y2, conf, cls_id])
        
        return detections
