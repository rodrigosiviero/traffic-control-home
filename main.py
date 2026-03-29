import sys
import os
import argparse
import logging
import time
import threading
from pathlib import Path

import cv2
import yaml
import numpy as np

from src.camera import RTSPCamera
from src.detector import VehicleDetector, COCO_VEHICLES
from src.tracker import VehicleTracker
from src.speed import SpeedEstimator
from src.direction import DirectionChecker
from src.alert import AlertManager
from src import metrics as metrics_module
from src.mqtt_client import MQTTPublisher
from src.api import APIServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("traffic-monitor")

IN_DOCKER = os.path.exists("/.dockerenv")
DATA_DIR = Path("/data" if IN_DOCKER else ".")

# Checar se OpenCV suporta GUI
_HAS_GUI = True
try:
    cv2.imshow("__test__", np.zeros((1, 1, 3), dtype=np.uint8))
    cv2.destroyAllWindows()
except cv2.error:
    _HAS_GUI = False


_DEBUG_MAX_WIDTH = 1280  # Largura máxima da janela debug


def _debug_show(window, frame, wait=1):
    """Mostra frame no debug se GUI disponível. Retorna True se 'q' pressionado."""
    if not _HAS_GUI:
        return False
    try:
        h, w = frame.shape[:2]
        if w > _DEBUG_MAX_WIDTH:
            scale = _DEBUG_MAX_WIDTH / w
            frame = cv2.resize(frame, (_DEBUG_MAX_WIDTH, int(h * scale)))
        cv2.imshow(window, frame)
        key = cv2.waitKey(wait) & 0xFF
        return key == ord("q")
    except cv2.error:
        return False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def test_connection(config: dict) -> bool:
    rtsp_url = config["camera"]["rtsp_url"]
    logger.info(f"Testando conexão: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error("Falha ao conectar RTSP")
        return False
    
    ret, frame = cap.read()
    if not ret:
        logger.error("Não conseguiu ler frame")
        cap.release()
        return False
    
    h, w = frame.shape[:2]
    logger.info(f"OK! Resolução: {w}x{h}")
    
    test_path = DATA_DIR / "logs" / "test_frame.jpg"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(test_path), frame)
    logger.info(f"Frame salvo: {test_path}")
    
    cap.release()
    return True


def run_monitor(config: dict, debug: bool = False, video_file: str = None, loop: bool = False, window_width: int = 1280):
    """Loop principal do monitoramento."""
    
    _VIDEO_FILE_MODE = video_file is not None
    
    if debug and not _HAS_GUI:
        logger.warning("OpenCV sem suporte a GUI! Rodando sem debug visual.")
        logger.warning("Instale 'opencv-python' (não headless) para usar --debug")
        debug = False
    
    if _VIDEO_FILE_MODE and debug:
        cv2.namedWindow("Traffic Monitor", cv2.WINDOW_NORMAL)
    
    # Pastas
    clips_dir = DATA_DIR / "clips"
    logs_dir = DATA_DIR / "logs"
    clips_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    alerts_cfg = config.get("alerts", {})
    alerts_cfg["clips_folder"] = str(clips_dir)
    alerts_cfg.setdefault("log_file", str(logs_dir / "alerts.log"))
    config["alerts"] = alerts_cfg
    
    # Componentes
    if _VIDEO_FILE_MODE:
        # Modo vídeo: abrir arquivo ao invés de RTSP
        video_cap = cv2.VideoCapture(video_file)
        if not video_cap.isOpened():
            logger.error(f"Não conseguiu abrir vídeo: {video_file}")
            return
        vid_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_fps = video_cap.get(cv2.CAP_PROP_FPS) or 25.0
        vid_total = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Modo vídeo: {video_file}")
        logger.info(f"Resolução: {vid_w}x{vid_h} | FPS: {vid_fps:.0f} | Frames: {vid_total}")
        
        # Criar objeto camera-like pra compatibilidade
        class VideoSource:
            def __init__(self, cap, path, loop_mode):
                self.cap = cap
                self.path = path
                self.loop = loop_mode
                self._native_width = vid_w
                self._native_height = vid_h
            def read(self):
                ret, frame = self.cap.read()
                if not ret and self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret:
                        logger.info("--- Vídeo em loop ---")
                return ret, frame
            def get_native_size(self):
                return self._native_width, self._native_height
            def reconnect(self):
                pass  # Não precisa reconectar arquivo
        
        camera = VideoSource(video_cap, video_file, loop)
    else:
        camera = RTSPCamera(config["camera"])
    
    detector = VehicleDetector(config["detection"])
    tracker = VehicleTracker(config.get("speed", {}))
    speed_estimator = SpeedEstimator(config["calibration"], config["speed"])
    direction_checker = DirectionChecker(config["direction"])
    alert_manager = AlertManager(config["alerts"])
    
    # API
    api_cfg = config.get("api", {})
    api_port = api_cfg.get("port", 8090)
    api_server = APIServer(api_port, DATA_DIR)
    api_server.start()
    
    # MQTT
    mqtt_cfg = config.get("mqtt", {})
    mqtt_publisher = MQTTPublisher(mqtt_cfg, DATA_DIR)
    
    # Injetar deps
    prom_enabled = config.get("prometheus", {}).get("enabled", True)
    alert_manager.set_deps(
        metrics_module if prom_enabled else None,
        mqtt_publisher,
        DATA_DIR,
    )
    
    metrics_module.set_camera_connected(False)
    
    logger.info("=" * 50)
    logger.info("Traffic Monitor iniciado")
    logger.info(f"  Modelo: YOLOv8{config['detection']['model_size']}")
    logger.info(f"  Classes: {detector.class_names}")
    logger.info(f"  Direção: {config['direction']['expected']}")
    logger.info(f"  Limite: {config['speed']['limit_kmh']} km/h")
    logger.info(f"  Data dir: {DATA_DIR}")
    logger.info(f"  Prometheus: {'ON' if prom_enabled else 'OFF'} (:{api_port}/metrics)")
    logger.info(f"  MQTT: {'ON' if mqtt_publisher.enabled else 'OFF'}")
    logger.info(f"  Debug: {'ON' if debug else 'OFF'}")
    logger.info("=" * 50)
    
    if not _VIDEO_FILE_MODE:
        if not camera.connect():
            logger.error("Não conseguiu conectar na câmera")
            return
        metrics_module.set_camera_connected(True)
        logger.info("Câmera conectada!")
    
    frame_count = 0
    skip_frames = 1 if _VIDEO_FILE_MODE else config["camera"].get("skip_frames", 3)
    
    fps_start = time.time()
    fps_count = 0
    current_fps = 0.0
    
    roi_polygon = config.get("roi", {}).get("polygon", [])
    
    display_frame = None
    
    status_interval = api_cfg.get("status_interval", 30)
    last_status_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                if _VIDEO_FILE_MODE:
                    if not loop:
                        logger.info("Fim do vídeo.")
                        break
                    # loop=True já trata no VideoSource.read()
                    continue
                if debug and display_frame is not None:
                    if _debug_show("Traffic Monitor", display_frame, 30):
                        break
                else:
                    logger.warning("Frame perdido, reconectando...")
                    metrics_module.set_camera_connected(False)
                    camera.reconnect()
                    metrics_module.set_camera_connected(True)
                continue
            
            frame_count += 1
            fps_count += 1
            
            alert_manager.add_frame(frame, frame_count)
            
            now = time.time()
            if now - fps_start >= 1.0:
                current_fps = fps_count / (now - fps_start)
                effective_fps = current_fps / max(skip_frames, 1)
                speed_estimator.set_fps(effective_fps)
                metrics_module.update_fps(current_fps)
                fps_count = 0
                fps_start = now
            
            # Status MQTT periódico
            if now - last_status_time >= status_interval:
                status = metrics_module.get_status_dict()
                mqtt_publisher.publish_status(status)
                last_status_time = now
            
            # Skip frames
            if frame_count % skip_frames != 0:
                if debug and display_frame is not None:
                    if _debug_show("Traffic Monitor", display_frame, 1):
                        break
                continue
            
            # === PROCESSAMENTO ===
            process_frame = frame  # Sem cópia, economia de RAM
            
            # No modo vídeo, limitar resolução (original pode ser 4K+ e estourar RAM)
            if _VIDEO_FILE_MODE:
                max_w = 1920
                fh, fw = process_frame.shape[:2]
                if fw > max_w:
                    ratio = max_w / fw
                    new_h = int(fh * ratio)
                    small_frame = cv2.resize(process_frame, (max_w, new_h))
                    pw, ph = max_w, new_h
                else:
                    small_frame = process_frame
                    pw, ph = fw, fh
            else:
                pw = config["camera"].get("process_width", 640)
                ph = config["camera"].get("process_height", 480)
                small_frame = cv2.resize(process_frame, (pw, ph))
            
            scale_x = frame.shape[1] / pw
            scale_y = frame.shape[0] / ph
            
            t0 = time.time()
            detections = detector.detect(small_frame)
            latency = time.time() - t0
            metrics_module.update_latency(latency)
            
            # Filtrar por ROI (pós-detecção) ao invés de mascarar o frame
            if not _VIDEO_FILE_MODE and roi_polygon and len(roi_polygon) >= 3:
                roi_pts = np.array(roi_polygon, dtype=np.float32)
                filtered = []
                for det in detections:
                    # Centro da bounding box
                    cx = (det[0] + det[2]) / 2.0
                    cy = (det[1] + det[3]) / 2.0
                    # Escalar pro espaço do frame (ROI coords = frame coords)
                    cx *= scale_x
                    cy *= scale_y
                    # Testar se ponto está dentro do polígono
                    if cv2.pointPolygonTest(roi_pts, (cx, cy), False) >= 0:
                        filtered.append(det)
                if debug and frame_count % 30 == 0:
                    logger.info(f"[DEBUG] ROI filtrou: {len(detections)} → {len(filtered)} detecções")
                detections = filtered
            
            if debug and frame_count % 30 == 0:
                logger.info(f"[DEBUG] Frame {frame_count}: {len(detections)} detecções, resolução: {pw}x{ph}")
            
            tracks = tracker.update(detections, frame_count)
            metrics_module.update_active_tracks(len(tracks))
            
            if debug:
                debug_frame = frame.copy()
                # Desenhar detecções cruas (amarelo) pra debug
                for det in detections:
                    dx1 = int(np.clip(det[0] * scale_x, 0, frame.shape[1]))
                    dy1 = int(np.clip(det[1] * scale_y, 0, frame.shape[0]))
                    dx2 = int(np.clip(det[2] * scale_x, 0, frame.shape[1]))
                    dy2 = int(np.clip(det[3] * scale_y, 0, frame.shape[0]))
                    cls_id = det[4]
                    conf = det[5]
                    cls_name = COCO_VEHICLES.get(cls_id, str(cls_id))
                    cv2.rectangle(debug_frame, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
                    cv2.putText(debug_frame, f"{cls_name} {conf:.0%}", (dx1, dy1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                debug_frame = None
            
            for track_id, track_data in tracks.items():
                history = track_data["history"]
                if len(history) < 2:
                    continue
                
                scaled_history = [(p[0] * scale_x, p[1] * scale_y) for p in history]
                
                direction_result = direction_checker.check(scaled_history)
                speed_kmh = speed_estimator.estimate(scaled_history)
                
                class_id = int(track_data["last_detection"][5])
                class_names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
                class_name = class_names.get(class_id, "unknown")
                
                if prom_enabled:
                    metrics_module.record_vehicle(class_id, speed_kmh)
                
                if mqtt_publisher.enabled and len(history) == 5:
                    mqtt_publisher.publish_vehicle(track_id, class_name, speed_kmh)
                
                is_wrong_way = direction_result["is_wrong_way"]
                is_speeding = (speed_kmh is not None and 
                              speed_kmh > config["speed"]["limit_kmh"] + 
                              config["speed"].get("tolerance_kmh", 5))
                
                if is_wrong_way:
                    alert_manager.alert(
                        alert_type="wrong_way",
                        track_id=track_id,
                        details={
                            "direction": direction_result["direction"],
                            "expected": direction_result["expected"],
                            "speed": speed_kmh,
                        },
                        frame=frame,
                    )
                
                if is_speeding:
                    alert_manager.alert(
                        alert_type="speeding",
                        track_id=track_id,
                        details={
                            "speed_kmh": speed_kmh,
                            "limit_kmh": config["speed"]["limit_kmh"],
                        },
                        frame=frame,
                    )
                
                # Debug overlay
                if debug and debug_frame is not None:
                    det = track_data["last_detection"]
                    x1 = int(np.clip(det[0] * scale_x, 0, frame.shape[1]))
                    y1 = int(np.clip(det[1] * scale_y, 0, frame.shape[0]))
                    x2 = int(np.clip(det[2] * scale_x, 0, frame.shape[1]))
                    y2 = int(np.clip(det[3] * scale_y, 0, frame.shape[0]))
                    
                    color = (0, 255, 0)
                    label = f"ID:{track_id} {class_name}"
                    
                    if is_wrong_way:
                        color = (0, 0, 255)
                        label += " CONTRARIO!"
                    elif is_speeding:
                        color = (0, 165, 255)
                        label += f" {speed_kmh:.0f}km/h!"
                    elif speed_kmh is not None:
                        label += f" {speed_kmh:.0f}km/h"
                    
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if debug and debug_frame is not None:
                # Desenhar ROI no debug
                if not _VIDEO_FILE_MODE and roi_polygon and len(roi_polygon) >= 3:
                    roi_pts = np.array(roi_polygon, dtype=np.int32)
                    cv2.polylines(debug_frame, [roi_pts], True, (255, 0, 255), 2)
                
                cv2.putText(debug_frame,
                           f"FPS: {current_fps:.1f} | Det: {len(detections)} | Tracks: {len(tracks)} | {latency*1000:.0f}ms",
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                display_frame = debug_frame
                if _debug_show("Traffic Monitor", display_frame, 1):
                    break
    
    except KeyboardInterrupt:
        logger.info("Parando...")
    finally:
        mqtt_publisher.disconnect()
        api_server.stop()
        if _VIDEO_FILE_MODE:
            video_cap.release()
        else:
            camera.disconnect()
            metrics_module.set_camera_connected(False)
        if debug and _HAS_GUI:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        logger.info("Monitoramento encerrado")


def main():
    parser = argparse.ArgumentParser(description="Traffic Monitor - CPU Edition")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--debug", action="store_true", help="Mostrar janela de debug")
    parser.add_argument("--test-connection", action="store_true", help="Testar RTSP")
    parser.add_argument("--file", default=None, help="Usar vídeo ao invés de RTSP (debug)")
    parser.add_argument("--loop", action="store_true", help="Repetir vídeo em loop (com --file)")
    parser.add_argument("--window-width", type=int, default=1280, help="Largura da janela debug")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        if os.path.exists("config.example.yaml"):
            logger.error(f"Config não encontrado: {args.config}")
            logger.error("  cp config.example.yaml config.yaml")
            sys.exit(1)
        else:
            logger.error(f"Config não encontrado: {args.config}")
            sys.exit(1)
    
    config = load_config(args.config)
    
    if args.test_connection:
        ok = test_connection(config)
        sys.exit(0 if ok else 1)
    
    run_monitor(config, debug=args.debug, video_file=args.file,
                loop=args.loop, window_width=args.window_width)


if __name__ == "__main__":
    main()
