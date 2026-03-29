import logging
import time

import cv2
import numpy as np

logger = logging.getLogger("traffic-monitor.camera")


class RTSPCamera:
    """Gerencia conexão RTSP com reconexão automática."""
    
    def __init__(self, config: dict):
        self.rtsp_url = config["rtsp_url"]
        self.process_width = config.get("process_width", 640)
        self.process_height = config.get("process_height", 480)
        self.cap = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
    
    def connect(self) -> bool:
        logger.info(f"Conectando em {self._mask_url()}...")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        if not self.cap.isOpened():
            logger.error("Falha ao abrir stream RTSP")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Conectou mas não conseguiu ler frame")
            self.cap.release()
            return False
        
        h, w = frame.shape[:2]
        logger.info(f"Conectado! Resolução nativa: {w}x{h}")
        self._reconnect_attempts = 0
        return True
    
    def read(self):
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame
    
    def reconnect(self):
        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._max_reconnect_attempts:
            logger.error(f"Máximo de tentativas atingido ({self._max_reconnect_attempts})")
            return False
        
        wait = min(self._reconnect_delay * self._reconnect_attempts, 60)
        logger.info(f"Reconectando em {wait}s (tentativa {self._reconnect_attempts})...")
        time.sleep(wait)
        self.disconnect()
        return self.connect()
    
    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def _mask_url(self) -> str:
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', self.rtsp_url)
