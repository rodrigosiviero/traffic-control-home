"""
Métricas Prometheus para o Traffic Monitor.

Expõe contadores, gauges e histogramas em /metrics.
"""
import logging
import time
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

logger = logging.getLogger("traffic-monitor.metrics")

# Registry customizado (não poluir o default)
registry = CollectorRegistry()

# === CONTADORES ===
vehicles_total = Counter(
    "traffic_vehicles_detected_total",
    "Total de veículos detectados",
    registry=registry,
)

wrong_way_total = Counter(
    "traffic_wrong_way_total",
    "Total de veículos na contramão",
    registry=registry,
)

speeding_total = Counter(
    "traffic_speeding_total",
    "Total de veículos acima do limite de velocidade",
    registry=registry,
)

speeding_by_bucket_total = Counter(
    "traffic_speeding_by_excess_bucket_total",
    "Veículos acima da velocidade por faixa de excesso",
    ["bucket"],
    registry=registry,
)

cars_total = Counter(
    "traffic_cars_total",
    "Total de carros detectados",
    registry=registry,
)

motorcycles_total = Counter(
    "traffic_motorcycles_total",
    "Total de motos detectadas",
    registry=registry,
)

trucks_total = Counter(
    "traffic_trucks_total",
    "Total de caminhões detectados",
    registry=registry,
)

buses_total = Counter(
    "traffic_buses_total",
    "Total de ônibus detectados",
    registry=registry,
)

# === GAUGES ===
current_speed_kmh = Gauge(
    "traffic_current_speed_kmh",
    "Velocidade do último veículo detectado (km/h)",
    registry=registry,
)

active_tracks = Gauge(
    "traffic_active_tracks",
    "Veículos sendo rastreados agora",
    registry=registry,
)

detection_fps = Gauge(
    "traffic_detection_fps",
    "FPS do processamento de detecção",
    registry=registry,
)

processing_latency_seconds = Gauge(
    "traffic_processing_latency_seconds",
    "Tempo de inferência YOLO por frame",
    registry=registry,
)

camera_connected = Gauge(
    "traffic_camera_connected",
    "Câmera RTSP conectada (1=sim, 0=não)",
    registry=registry,
)

uptime_seconds = Gauge(
    "traffic_uptime_seconds",
    "Tempo de atividade do monitor",
    registry=registry,
)

# === HISTOGRAMA ===
speed_distribution = Histogram(
    "traffic_vehicle_speed_kmh",
    "Distribuição de velocidades dos veículos",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150],
    registry=registry,
)

# Mapeamento de classes
CLASS_COUNTERS = {
    2: cars_total,       # car
    3: motorcycles_total, # motorcycle
    5: buses_total,      # bus
    7: trucks_total,     # truck
}

# Start time
_start_time = time.time()


def record_vehicle(class_id: int, speed: float | None):
    """Registra um veículo detectado."""
    vehicles_total.inc()
    
    if class_id in CLASS_COUNTERS:
        CLASS_COUNTERS[class_id].inc()
    
    if speed is not None:
        current_speed_kmh.set(speed)
        speed_distribution.observe(speed)


def record_wrong_way(speed: float | None = None):
    """Registra infração de contramão."""
    wrong_way_total.inc()


def record_speeding(speed_kmh: float, limit_kmh: float):
    """Registra infração de velocidade."""
    speeding_total.inc()
    
    excess = speed_kmh - limit_kmh
    # Bucket por faixa de excesso
    if excess <= 10:
        bucket = "0-10"
    elif excess <= 20:
        bucket = "10-20"
    elif excess <= 30:
        bucket = "20-30"
    else:
        bucket = "30+"
    speeding_by_bucket_total.labels(bucket=bucket).inc()


def update_active_tracks(count: int):
    active_tracks.set(count)


def update_fps(fps: float):
    detection_fps.set(fps)


def update_latency(seconds: float):
    processing_latency_seconds.set(seconds)


def set_camera_connected(connected: bool):
    camera_connected.set(1 if connected else 0)


def update_uptime():
    uptime_seconds.set(time.time() - _start_time)


def get_metrics() -> bytes:
    """Retorna métricas no formato Prometheus."""
    update_uptime()
    return generate_latest(registry)


def get_metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


# === HELPERS para status MQTT ===
def get_status_dict() -> dict:
    """Retorna dict com valores atuais das métricas pra enviar via MQTT."""
    update_uptime()
    return {
        "uptime_seconds": round(uptime_seconds._value.get(), 0),
        "camera_connected": bool(camera_connected._value.get()),
        "fps": round(detection_fps._value.get(), 1),
        "active_tracks": int(active_tracks._value.get()),
        "vehicles_total": int(vehicles_total._value.get()),
        "wrong_way_total": int(wrong_way_total._value.get()),
        "speeding_total": int(speeding_total._value.get()),
        "last_speed_kmh": round(current_speed_kmh._value.get(), 1),
    }
