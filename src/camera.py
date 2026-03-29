import logging
import time
import re

import cv2

logger = logging.getLogger("traffic-monitor.camera")


class RTSPCamera:
    """
    Gerencia conexão RTSP — leitura síncrona.
    
    O RTSP já é bloqueante (espera próximo frame). 
    Não precisa de thread dedicada.
    """
    
    def __init__(self, config: dict):
        self.rtsp_url = config["rtsp_url"]
        self.cap = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
        self._connected = False
    
    def connect(self) -> bool:
        logger.info(f"Conectando em {self._mask_url()}...")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            logger.error("Falha ao abrir stream RTSP")
            return False
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Stream abriu mas não leu frame")
            self.cap.release()
            return False
        
        self._native_width = frame.shape[1]
        self._native_height = frame.shape[0]
        self._connected = True
        logger.info(f"Conectado! Nativo: {self._native_width}x{self._native_height}")
        
        return True
    
    def read(self):
        """Lê próximo frame (bloqueia até chegar)."""
        ret, frame = self.cap.read()
        if ret:
            self._connected = True
        return ret, frame
    
    def get_native_size(self):
        return getattr(self, '_native_width', 0), getattr(self, '_native_height', 0)
    
    def reconnect(self):
        """Reconecta com backoff progressivo."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._reconnect_attempts = 0
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = min(self._reconnect_delay * self._reconnect_attempts, 60)
            logger.warning(f"Reconectando ({self._reconnect_attempts}/{self._max_reconnect_attempts}) em {delay}s...")
            time.sleep(delay)
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self._native_width = frame.shape[1]
                    self._native_height = frame.shape[0]
                    self._reconnect_attempts = 0
                    self._connected = True
                    logger.info(f"Reconectado! ({self._native_width}x{self._native_height})")
                    return
        
        logger.error(f"Falha após {self._max_reconnect_attempts} tentativas")
    
    def disconnect(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._connected = False
        logger.info("Câmera desconectada")
    
    def _mask_url(self):
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', self.rtsp_url)
