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
    """
    Detecção de veículos usando YOLO.
    
    Suporta 3 backends (em ordem de preferência):
    1. openvino + Intel GPU  → mais rápido, usa iGPU
    2. openvino + CPU        → mais rápido que PyTorch
    3. pytorch + CPU         → fallback universal
    """
    
    def __init__(self, config: dict):
        self.model_size = config.get("model_size", "n")
        self.confidence = config.get("confidence", 0.3)
        self.target_classes = config.get("classes", [2, 3, 5, 7])
        self.imgsz = config.get("imgsz", 640)
        self.device = config.get("device", "auto")  # "auto", "gpu", "cpu"
        
        self.class_names = {k: COCO_VEHICLES[k] for k in self.target_classes if k in COCO_VEHICLES}
        
        self.model = None
        self.backend = "unknown"
        self._load_model()
    
    def _load_model(self):
        """Carrega modelo YOLO com melhor backend disponível."""
        from ultralytics import YOLO
        
        model_name = f"yolov8{self.model_size}"
        
        # Tentar OpenVINO primeiro (mais rápido em CPU e muito mais rápido em iGPU)
        if self._try_openvino(YOLO, model_name):
            return
        
        # Fallback: PyTorch CPU
        self._load_pytorch(YOLO, model_name)
    
    def _try_openvino(self, YOLO, model_name: str) -> bool:
        """Tenta carregar modelo OpenVINO (exporta se necessário)."""
        try:
            import openvino  # noqa: F401
        except ImportError:
            logger.info("OpenVINO não instalado — usando PyTorch")
            return False
        
        # Verificar se modelo exportado já existe
        model_dir = Path(f"{model_name}_openvino_model")
        model_file = model_dir / "best.xml" if model_dir.exists() else None
        
        if model_file is None:
            model_file = Path(f"{model_name}_openvino_model.xml")
        
        if not model_file or not model_file.exists():
            logger.info("Exportando modelo para OpenVINO (primeira vez)...")
            try:
                pt_model = YOLO(f"{model_name}.pt")
                pt_model.export(
                    format="openvino",
                    imgsz=self.imgsz,
                    half=False,  # FP32 — mais compatível com Intel iGPU
                )
                logger.info("Exportação OpenVINO concluída!")
            except Exception as e:
                logger.warning(f"Falha ao exportar OpenVINO: {e}")
                return False
        
        # Determinar device
        ov_device = self._resolve_openvino_device()
        
        try:
            # Ultralytics >= 8.1 carrega OpenVINO nativamente
            model_path = model_dir if model_dir.exists() else model_file
            self.model = YOLO(str(model_path), task="detect")
            
            # Warmup — primeiro predict inicializa o backend
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, imgsz=self.imgsz)
            
            self.backend = f"openvino/{ov_device}"
            logger.info(f"✓ Modelo carregado: OpenVINO no {ov_device}")
            return True
            
        except Exception as e:
            logger.warning(f"Falha ao carregar OpenVINO ({ov_device}): {e}")
            # Tentar CPU como fallback dentro do OpenVINO
            if ov_device != "CPU":
                try:
                    self.model = YOLO(str(model_path), task="detect")
                    dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                    self.model.predict(dummy, verbose=False, imgsz=self.imgsz)
                    self.backend = "openvino/CPU"
                    logger.info("✓ Modelo carregado: OpenVINO no CPU (GPU falhou)")
                    return True
                except Exception as e2:
                    logger.warning(f"OpenVINO CPU também falhou: {e2}")
            return False
    
    def _resolve_openvino_device(self) -> str:
        """Determina qual dispositivo OpenVINO usar."""
        if self.device == "cpu":
            return "CPU"
        
        if self.device == "gpu":
            return "GPU"
        
        # Auto: tentar GPU primeiro
        try:
            from openvino.runtime import Core
            core = Core()
            devices = core.available_devices
            logger.info(f"OpenVINO devices disponíveis: {devices}")
            
            if "GPU" in devices:
                # Testar se a GPU realmente funciona
                try:
                    # Query básico pra verificar se /dev/dri está acessível
                    core.get_property("GPU", "FULL_DEVICE_NAME")
                    return "GPU"
                except Exception as e:
                    logger.warning(f"Intel GPU encontrada mas inacessível: {e}")
                    logger.info("Dica: adicione 'devices: [/dev/dri:/dev/dri]' no docker-compose")
                    return "CPU"
            return "CPU"
        except Exception:
            return "CPU"
    
    def _load_pytorch(self, YOLO, model_name: str):
        """Fallback: carrega modelo PyTorch em CPU."""
        model_file = f"{model_name}.pt"
        self.model = YOLO(model_file)
        
        # Warmup
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, imgsz=self.imgsz)
        
        self.backend = "pytorch/cpu"
        logger.info(f"✓ Modelo carregado: PyTorch CPU")
    
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
            device=self._get_predict_device(),
            half=False,
            max_det=20,
            imgsz=self.imgsz,
        )
        
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        
        # Extração vetorizada
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clsids = boxes.cls.cpu().numpy()
        
        return np.column_stack([xyxy, confs[:, None], clsids[:, None]]).tolist()
    
    def _get_predict_device(self) -> str:
        """Retorna o device correto pra model.predict()."""
        if "openvino" in self.backend:
            return ""  # OpenVINO gerencia o device internamente
        return "cpu"
