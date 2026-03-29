import logging
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from collections import deque

import cv2
import numpy as np

logger = logging.getLogger("traffic-monitor.alert")


class AlertManager:
    """Gerencia alertas de infrações de trânsito."""
    
    def __init__(self, config: dict):
        self.log_file = config.get("log_file", "logs/alerts.log")
        self.save_clips = config.get("save_clips", True)
        self.clips_folder = Path(config.get("clips_folder", "clips"))
        self.clip_duration = config.get("clip_duration_sec", 5)
        self.terminal_beep = config.get("terminal_beep", True)
        self.cooldown_sec = config.get("cooldown_sec", 30)
        self.webhook_url = config.get("webhook_url")
        
        self.clips_folder.mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        
        self._last_alert = {}
        
        # Buffer circular de frames
        self.buffer_secs = self.clip_duration
        self.after_secs = self.clip_duration
        self._frame_buffer = deque(maxlen=300)
        self._buffer_lock = threading.Lock()
        
        self._active_recordings = {}
        self._rec_lock = threading.Lock()
        
        # Externos (setados pelo main.py)
        self.metrics = None       # src.metrics
        self.mqtt = None          # src.mqtt_client.MQTTPublisher
        self.data_dir = Path(".")
        
        # Últimos arquivos salvos (para MQTT saber o path)
        self._last_photo = None
        self._last_clip = None
    
    def set_deps(self, metrics_module, mqtt_publisher, data_dir: Path):
        """Injeta dependências externas."""
        self.metrics = metrics_module
        self.mqtt = mqtt_publisher
        self.data_dir = data_dir
    
    def add_frame(self, frame, frame_num: int):
        if frame is None:
            return
        with self._buffer_lock:
            self._frame_buffer.append({
                "frame": frame.copy(),
                "frame_num": frame_num,
                "timestamp": time.time(),
            })
    
    def alert(self, alert_type: str, track_id: int, details: dict, frame: np.ndarray):
        now = time.time()
        
        # Cooldown por track ID (evita alertas repetidos do mesmo track)
        key = (alert_type, track_id)
        if key in self._last_alert:
            if now - self._last_alert[key] < self.cooldown_sec:
                return
        
        # Cooldown global por tipo (evita 2 alertas do mesmo carro com IDs diferentes)
        global_key = (alert_type, "global")
        if global_key in self._last_alert:
            if now - self._last_alert[global_key] < self.cooldown_sec:
                return
        
        self._last_alert[key] = now
        self._last_alert[global_key] = now
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if alert_type == "wrong_way":
            msg = (f"[{timestamp}] CONTRAMÃO! Veículo #{track_id} "
                   f"direção: {details['direction']}, esperada: {details['expected']}")
            if details.get("speed"):
                msg += f", velocidade: {details['speed']:.0f} km/h"
        elif alert_type == "speeding":
            msg = (f"[{timestamp}] EXCESSO VELOCIDADE! Veículo #{track_id} "
                   f"{details['speed_kmh']:.0f} km/h (limite: {details['limit_kmh']} km/h)")
        else:
            msg = f"[{timestamp}] Alerta {alert_type}: veículo #{track_id}"
        
        logger.warning(msg)
        self._write_log(msg)
        
        if self.terminal_beep:
            print("\a", end="", flush=True)
        
        # Screenshot
        photo_path = None
        if frame is not None:
            photo_path = self._save_screenshot(alert_type, track_id, timestamp, frame, details)
        
        # Gravar clip
        clip_path = None
        if self.save_clips and frame is not None:
            clip_path = self._record_clip(alert_type, track_id, timestamp, frame, details)
        
        # === MÉTRICAS PROMETHEUS ===
        if self.metrics:
            if alert_type == "wrong_way":
                self.metrics.record_wrong_way(details.get("speed"))
            elif alert_type == "speeding":
                self.metrics.record_speeding(details["speed_kmh"], details["limit_kmh"])
        
        # === MQTT ===
        if self.mqtt:
            if alert_type == "wrong_way":
                self.mqtt.publish_wrong_way(track_id, details, photo_path, clip_path)
            elif alert_type == "speeding":
                self.mqtt.publish_speeding(track_id, details, photo_path, clip_path)
        
        # Webhook
        if self.webhook_url:
            self._send_webhook(alert_type, track_id, details, timestamp, photo_path, clip_path)
    
    def _record_clip(self, alert_type, track_id, timestamp, trigger_frame, details):
        """Grava clip e retorna o path do arquivo."""
        with self._rec_lock:
            if track_id in self._active_recordings:
                return self._last_clip
            self._active_recordings[track_id] = True
        
        ts_clean = timestamp.replace(":", "-").replace(" ", "_")
        filename = f"{alert_type}_id{track_id}_{ts_clean}.mp4"
        filepath = self.clips_folder / filename
        
        self._last_clip = str(filepath)
        
        t = threading.Thread(
            target=self._do_record_clip,
            args=(alert_type, track_id, timestamp, trigger_frame.copy(), details, filepath),
            daemon=True,
        )
        t.start()
        
        return str(filepath)
    
    def _do_record_clip(self, alert_type, track_id, timestamp, trigger_frame, details, filepath):
        try:
            with self._buffer_lock:
                buffered = list(self._frame_buffer)
            
            if not buffered:
                logger.warning("Buffer vazio, pulando clip")
                with self._rec_lock:
                    del self._active_recordings[track_id]
                return
            
            h, w = buffered[0]["frame"].shape[:2]
            
            trigger_time = time.time()
            cutoff_time = trigger_time - self.buffer_secs
            
            retro_frames = [f for f in buffered if f["timestamp"] >= cutoff_time]
            if not retro_frames:
                retro_frames = [buffered[-1]]
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = 15.0
            writer = cv2.VideoWriter(str(filepath), fourcc, fps_out, (w, h))
            
            if not writer.isOpened():
                logger.error(f"Falha ao criar vídeo: {filepath}")
                with self._rec_lock:
                    del self._active_recordings[track_id]
                return
            
            alert_label = "CONTRAMAO!" if alert_type == "wrong_way" else \
                         f"EXCESSO: {details.get('speed_kmh', 0):.0f} km/h"
            alert_color = (0, 0, 255) if alert_type == "wrong_way" else (0, 165, 255)
            
            for f_data in retro_frames:
                annotated = self._annotate_frame(
                    f_data["frame"], alert_label, alert_color,
                    f_data["timestamp"], phase="PRE"
                )
                writer.write(annotated)
            
            after_end = time.time() + self.after_secs
            last_frame_time = retro_frames[-1]["timestamp"] if retro_frames else trigger_time
            
            while time.time() < after_end:
                time.sleep(0.1)
                with self._buffer_lock:
                    new_frames = [f for f in self._frame_buffer if f["timestamp"] > last_frame_time]
                for f_data in new_frames:
                    annotated = self._annotate_frame(
                        f_data["frame"], alert_label, alert_color,
                        f_data["timestamp"], phase="POS"
                    )
                    writer.write(annotated)
                    last_frame_time = f_data["timestamp"]
            
            writer.release()
            logger.info(f"Clip salvo: {filepath}")
        
        except Exception as e:
            logger.error(f"Erro ao gravar clip: {e}")
        finally:
            with self._rec_lock:
                self._active_recordings.pop(track_id, None)
    
    def _annotate_frame(self, frame, label, color, frame_time, phase=""):
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        
        cv2.putText(annotated, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        ts_str = datetime.fromtimestamp(frame_time).strftime("%H:%M:%S.%f")[:-3]
        phase_str = f"[{phase}]" if phase else ""
        cv2.putText(annotated, f"{ts_str} {phase_str}", (10, 48),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        return annotated
    
    def _write_log(self, message: str):
        try:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")
        except Exception as e:
            logger.error(f"Erro ao escrever log: {e}")
    
    def _save_screenshot(self, alert_type, track_id, timestamp, frame, details):
        try:
            ts_clean = timestamp.replace(":", "-").replace(" ", "_")
            filename = f"{alert_type}_id{track_id}_{ts_clean}.jpg"
            filepath = self.clips_folder / filename
            
            annotated = frame.copy()
            color = (0, 0, 255) if alert_type == "wrong_way" else (0, 165, 255)
            
            label = "CONTRAMAO!" if alert_type == "wrong_way" else \
                    f"EXCESSO: {details.get('speed_kmh', 0):.0f} km/h"
            cv2.putText(annotated, label, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(annotated, timestamp, (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imwrite(str(filepath), annotated)
            self._last_photo = str(filepath)
            logger.info(f"Screenshot salvo: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Erro ao salvar screenshot: {e}")
            return None
    
    def _send_webhook(self, alert_type, track_id, details, timestamp, photo_path=None, clip_path=None):
        try:
            import urllib.request
            payload = json.dumps({
                "alert_type": alert_type,
                "track_id": track_id,
                "details": details,
                "timestamp": timestamp,
                "photo_path": photo_path,
                "clip_path": clip_path,
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                logger.debug(f"Webhook: {resp.status}")
        except Exception as e:
            logger.error(f"Erro webhook: {e}")
