import logging
import numpy as np
import cv2

logger = logging.getLogger("traffic-monitor.speed")


class SpeedEstimator:
    """
    Estimativa de velocidade usando HOMOGRAFIA.
    
    Mapeia posições em pixels → posições no mundo real (metros)
    usando uma matriz de homografia calibrada. Isso corrige
    distorção de perspectiva da câmera.
    """
    
    def __init__(self, calibration: dict, speed_config: dict):
        self.limit_kmh = speed_config.get("limit_kmh", 40)
        self.tolerance_kmh = speed_config.get("tolerance_kmh", 5)
        self.min_track_frames = speed_config.get("min_track_frames", 3)
        
        self.fps = 10.0  # Será atualizado pelo loop principal
        
        # Detectar método de calibração
        method = calibration.get("method", "linear")
        
        if method == "homography" and "homography_matrix" in calibration:
            self._init_homography(calibration)
        else:
            self._init_linear(calibration)
    
    def _init_homography(self, calibration: dict):
        """Inicializa com matriz de homografia."""
        self.method = "homography"
        self.H = np.array(calibration["homography_matrix"], dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        
        n_points = len(calibration.get("points", []))
        logger.info(f"Calibração: HOMOGRAFIA ({n_points} pontos)")
        logger.info(f"Matriz H:\n{self.H}")
        
        # Validar: transformar um ponto de referência e ver se bate
        points = calibration.get("points", [])
        if points:
            test = points[0]
            px = np.array([[[test["pixel"][0], test["pixel"][1]]]], dtype=np.float32)
            real = cv2.perspectiveTransform(px, self.H)
            logger.info(f"Validação: pixel {test['pixel']} → real ({real[0][0][0]:.2f}, {real[0][0][1]:.2f})m (esperado: {test['real']})")
    
    def _init_linear(self, calibration: dict):
        """Fallback: calibração linear simples (2 pontos)."""
        self.method = "linear"
        
        point_a = calibration.get("point_a", [100, 400])
        point_b = calibration.get("point_b", [500, 400])
        real_distance = calibration.get("real_distance_m", 15.0)
        
        pixel_distance = np.sqrt(
            (point_b[0] - point_a[0]) ** 2 + 
            (point_b[1] - point_a[1]) ** 2
        )
        
        self.pixels_per_meter = pixel_distance / real_distance if real_distance > 0 else 30.0
        logger.info(f"Calibração: LINEAR ({self.pixels_per_meter:.1f} pixels/metro)")
    
    def pixel_to_real(self, px_x: float, px_y: float) -> tuple:
        """
        Converte coordenada pixel → coordenada real (metros).
        
        Com homografia: usa a matriz H.
        Linear: retorna distância em pixels / pixels_per_meter.
        """
        if self.method == "homography":
            pt = np.array([[[px_x, px_y]]], dtype=np.float32)
            real = cv2.perspectiveTransform(pt, self.H)
            return float(real[0][0][0]), float(real[0][0][1])
        else:
            # Linear: retorna posição em "metros" assumindo escala uniforme
            return px_x / self.pixels_per_meter, px_y / self.pixels_per_meter
    
    def real_to_pixel(self, real_x: float, real_y: float) -> tuple:
        """Converte coordenada real → pixel."""
        if self.method == "homography":
            pt = np.array([[[real_x, real_y]]], dtype=np.float32)
            px = cv2.perspectiveTransform(pt, self.H_inv)
            return float(px[0][0][0]), float(px[0][0][1])
        else:
            return real_x * self.pixels_per_meter, real_y * self.pixels_per_meter
    
    def estimate_speed(self, history: list) -> tuple:
        """
        Estima velocidade a partir do histórico de posições.
        
        Args:
            history: [(pixel_x, pixel_y), ...] posições do tracker
        
        Returns:
            (speed_kmh, is_speeding) ou (None, False)
        """
        if len(history) < self.min_track_frames:
            return None, False
        
        # Converter todo o histórico pra coordenadas reais
        real_history = [self.pixel_to_real(p[0], p[1]) for p in history]
        
        # Calcular distância total percorrida (em metros)
        total_dist = 0.0
        for i in range(1, len(real_history)):
            dx = real_history[i][0] - real_history[i-1][0]
            dy = real_history[i][1] - real_history[i-1][1]
            total_dist += np.sqrt(dx*dx + dy*dy)
        
        # Tempo = frames / fps
        n_intervals = len(history) - 1
        if n_intervals <= 0:
            return None, False
        
        elapsed_sec = n_intervals / max(self.fps, 1.0)
        
        if elapsed_sec < 0.1:  # Muito pouco tempo
            return None, False
        
        # Velocidade em m/s → km/h
        speed_ms = total_dist / elapsed_sec
        speed_kmh = speed_ms * 3.6
        
        is_speeding = speed_kmh > (self.limit_kmh + self.tolerance_kmh)
        
        return speed_kmh, is_speeding
    
    def set_fps(self, fps: float):
        """Atualiza FPS real do processamento."""
        self.fps = max(fps, 1.0)
