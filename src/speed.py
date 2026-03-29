import logging
import numpy as np
import cv2

logger = logging.getLogger("traffic-monitor.speed")


class SpeedEstimator:
    """
    Estimativa de velocidade usando HOMOGRAFIA + timestamps reais.
    
    Cada ponto do histórico do tracker tem (x, y, timestamp).
    A homografia converte pixels → metros, e os timestamps
    dão o tempo real. Sem depender de FPS estimado.
    """
    
    def __init__(self, calibration: dict, speed_config: dict):
        self.limit_kmh = speed_config.get("limit_kmh", 40)
        self.tolerance_kmh = speed_config.get("tolerance_kmh", 5)
        self.min_track_frames = speed_config.get("min_track_frames", 3)
        
        # Detectar método de calibração
        method = calibration.get("method", "linear")
        
        if method == "homography" and "homography_matrix" in calibration:
            self._init_homography(calibration)
        else:
            self._init_linear(calibration)
    
    def _init_homography(self, calibration: dict):
        self.method = "homography"
        self.H = np.array(calibration["homography_matrix"], dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        
        n_points = len(calibration.get("points", []))
        logger.info(f"Speed: HOMOGRAFIA ({n_points} pontos)")
        
        # Validar
        points = calibration.get("points", [])
        if len(points) >= 2:
            # Distância P1→P2 em metros (via homografia)
            p1 = np.array([[[points[0]["pixel"][0], points[0]["pixel"][1]]]], dtype=np.float32)
            p2 = np.array([[[points[1]["pixel"][0], points[1]["pixel"][1]]]], dtype=np.float32)
            r1 = cv2.perspectiveTransform(p1, self.H)[0][0]
            r2 = cv2.perspectiveTransform(p2, self.H)[0][0]
            dist = np.sqrt((r2[0]-r1[0])**2 + (r2[1]-r1[1])**2)
            expected = np.sqrt((points[1]["real"][0]-points[0]["real"][0])**2 +
                              (points[1]["real"][1]-points[0]["real"][1])**2)
            logger.info(f"  Validação P1→P2: {dist:.2f}m (esperado: {expected:.2f}m)")
    
    def _init_linear(self, calibration: dict):
        self.method = "linear"
        
        points = calibration.get("points", [])
        if len(points) >= 2:
            point_a = points[0]["pixel"]
            point_b = points[1]["pixel"]
            real_distance = np.sqrt(
                (points[1]["real"][0] - points[0]["real"][0])**2 +
                (points[1]["real"][1] - points[0]["real"][1])**2
            )
        elif "pixels_per_meter" in calibration:
            self.pixels_per_meter = calibration["pixels_per_meter"]
            logger.info(f"Speed: LINEAR ({self.pixels_per_meter:.1f} px/m)")
            return
        else:
            self.pixels_per_meter = 30.0
            logger.warning("Sem calibração, usando 30 px/m")
            return
        
        pixel_distance = np.sqrt(
            (point_b[0] - point_a[0]) ** 2 +
            (point_b[1] - point_a[1]) ** 2
        )
        self.pixels_per_meter = pixel_distance / real_distance if real_distance > 0 else 30.0
        logger.info(f"Speed: LINEAR ({self.pixels_per_meter:.1f} px/m)")
    
    def set_fps(self, fps: float):
        """Manter compatibilidade — não é mais usado para cálculo."""
        pass
    
    def _pixels_to_meters_batch(self, pixels: list) -> np.ndarray:
        """
        Converte lista de (x, y) pixels → array (N, 2) em metros.
        Uma única chamada perspectiveTransform.
        """
        arr = np.array(pixels, dtype=np.float32).reshape(1, -1, 2)
        if self.method == "homography":
            real = cv2.perspectiveTransform(arr, self.H)
            return real[0]  # (N, 2)
        else:
            return arr[0] / self.pixels_per_meter  # (N, 2)
    
    def estimate_speed(self, history: list) -> tuple:
        """
        Estima velocidade usando timestamps reais do tracker.
        
        Args:
            history: [(x, y, timestamp), ...] do tracker
        
        Returns:
            (speed_kmh, is_speeding) ou (None, False)
        """
        if len(history) < self.min_track_frames:
            return None, False
        
        # Extrair pixels e timestamps
        pixels = [(p[0], p[1]) for p in history]
        timestamps = [p[2] for p in history]
        
        # Tempo real entre primeiro e último ponto
        elapsed_sec = timestamps[-1] - timestamps[0]
        
        if elapsed_sec < 0.1:  # Menos de 100ms — impreciso
            return None, False
        
        # Converter pixels → metros (batch)
        real_pts = self._pixels_to_meters_batch(pixels)  # (N, 2) em metros
        
        # Distância total percorrida
        diffs = np.diff(real_pts, axis=0)  # (N-1, 2)
        segment_dists = np.sqrt((diffs ** 2).sum(axis=1))
        total_dist = segment_dists.sum()
        
        # Velocidade
        speed_ms = total_dist / elapsed_sec
        speed_kmh = speed_ms * 3.6
        
        is_speeding = speed_kmh > (self.limit_kmh + self.tolerance_kmh)
        
        # Log de debug (primeira vez, ou velocidade absurda)
        if speed_kmh > 200:
            logger.warning(f"Velocidade suspeita: {speed_kmh:.1f} km/h "
                          f"({total_dist:.2f}m em {elapsed_sec:.2f}s, "
                          f"{len(history)} pontos)")
        
        return speed_kmh, is_speeding
    
    def real_to_pixel(self, real_x: float, real_y: float) -> tuple:
        if self.method == "homography":
            pt = np.array([[[real_x, real_y]]], dtype=np.float32)
            px = cv2.perspectiveTransform(pt, self.H_inv)
            return float(px[0][0][0]), float(px[0][0][1])
        else:
            return real_x * self.pixels_per_meter, real_y * self.pixels_per_meter
