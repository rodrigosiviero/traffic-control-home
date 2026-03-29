import logging
import time
import threading

import cv2
import numpy as np

logger = logging.getLogger("traffic-monitor.camera")


class RTSPCamera:
    """Gerencia conexão RTSP com thread dedicada de captura."""
    
    def __init__(self, config: dict):
        self.rtsp_url = config["rtsp_url"]
        self.process_width = config.get("process_width", 640)
        self.process_height = config.get("process_height", 480)
        self.cap = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5
        
        # Thread dedicada de captura
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
    
    def connect(self) -> bool:
        logger.info(f"Conectando em {self._mask_url()}...")
        
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        
        if not self.cap.isOpened():
            logger.error("Falha ao abrir stream RTSP")
            return False
        
        # Ler primeiro frame pra confirmar
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Stream abriu mas não leu frame")
            self.cap.release()
            return False
        
        self._native_width = frame.shape[1]
        self._native_height = frame.shape[0]
        logger.info(f"Conectado! Resolução nativa: {self._native_width}x{self._native_height}")
        
        with self._lock:
            self._frame = frame
        
        # Iniciar thread de captura
        self._running = True
        self._connected = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def _capture_loop(self):
        """Thread dedicada — sempre lê o frame mais recente."""
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Frame perdido na captura")
                self._connected = False
                break
            with self._lock:
                self._frame = frame
            # Pequena pausa pra não girar em busy-loop
            time.sleep(0.01)
    
    def read(self):
        """Retorna o frame mais recente (não bloqueia)."""
        with self._lock:
            if self._frame is None:
                return False, None
            # Copy pra garantir que a thread não sobrescreva
            # enquanto o caller processa
            return True, self._frame.copy()
    
    def read_no_copy(self):
        """Retorna referência direta — usar só se não modificar o frame."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame
    
    def get_native_size(self):
        return getattr(self, '_native_width', 0), getattr(self, '_native_height', 0)
    
    def reconnect(self):
        """Reconecta com backoff progressivo."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        while self._reconnect_attempts < self._max_reconnect_attempts:
            self._reconnect_attempts += 1
            delay = min(self._reconnect_delay * self._reconnect_attempts, 60)
            logger.warning(f"Reconectando (tentativa {self._reconnect_attempts}/{self._max_reconnect_attempts}) em {delay}s...")
            time.sleep(delay)
            
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self._native_width = frame.shape[1]
                    self._native_height = frame.shape[0]
                    with self._lock:
                        self._frame = frame
                    self._reconnect_attempts = 0
                    self._connected = True
                    self._running = True
                    self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                    self._thread.start()
                    logger.info(f"Reconectado! ({self._native_width}x{self._native_height})")
                    return
                else:
                    self.cap.release()
        
        logger.error(f"Falha após {self._max_reconnect_attempts} tentativas")
    
    def disconnect(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._connected = False
        logger.info("Câmera desconectada")
    
    def _mask_url(self):
        """Mascara senha na URL pra logs."""
        import re
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:****@', self.rtsp_url)
