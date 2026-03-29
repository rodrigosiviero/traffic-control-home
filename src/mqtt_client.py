"""
Cliente MQTT para publicar eventos e status no Home Assistant.

Tópicos:
  {prefix}/status          → Status periódico (online, stats)
  {prefix}/wrong-way       → Evento de contramão
  {prefix}/speeding        → Evento de excesso de velocidade
  {prefix}/vehicle         → Todo veículo detectado (speed + class)
  
  {prefix}/discovery/{sensor_id}/config → Auto-discovery do HA
"""
import logging
import json
import time
import threading
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("traffic-monitor.mqtt")


class MQTTPublisher:
    """Publica eventos de trânsito via MQTT."""
    
    def __init__(self, config: dict, data_dir: Path):
        self.enabled = config.get("enabled", False)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 1883)
        self.username = config.get("username", "")
        self.password = config.get("password", "")
        self.prefix = config.get("topic_prefix", "traffic-monitor")
        self.data_dir = data_dir
        
        self._client = None
        self._connected = False
        self._lock = threading.Lock()
        
        if not self.enabled:
            logger.info("MQTT desabilitado")
            return
        
        self._connect()
    
    def _connect(self):
        """Conecta ao broker MQTT."""
        try:
            import paho.mqtt.client as mqtt
            
            self._client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id="traffic-monitor",
            )
            
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            
            # Last Will — marca offline se cair
            self._client.will_set(
                f"{self.prefix}/status",
                payload=json.dumps({"online": False}),
                qos=1,
                retain=True,
            )
            
            logger.info(f"Conectando MQTT: {self.host}:{self.port}...")
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            
        except Exception as e:
            logger.error(f"Falha ao conectar MQTT: {e}")
            self.enabled = False
    
    def _on_connect(self, client, userdata, flags, rc, properties=None):
        self._connected = True
        logger.info("MQTT conectado!")
        self.publish_status({"online": True})
        self._setup_home_assistant_discovery()
    
    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        self._connected = False
        if rc != 0:
            logger.warning(f"MQTT desconectou inesperadamente (rc={rc})")
    
    def _publish(self, topic_suffix: str, payload: dict, retain: bool = False):
        """Publica mensagem MQTT (thread-safe)."""
        if not self.enabled or not self._connected:
            return
        
        topic = f"{self.prefix}/{topic_suffix}"
        
        with self._lock:
            try:
                self._client.publish(
                    topic,
                    payload=json.dumps(payload, default=str),
                    qos=1,
                    retain=retain,
                )
            except Exception as e:
                logger.error(f"Erro MQTT publish: {e}")
    
    def publish_status(self, status: dict):
        """Publica status periódico."""
        status["online"] = True
        status["timestamp"] = datetime.now().isoformat()
        self._publish("status", status, retain=True)
    
    def publish_wrong_way(self, track_id: int, details: dict, photo_path: str = None, clip_path: str = None):
        """Publica evento de contramão."""
        payload = {
            "event": "wrong_way",
            "track_id": track_id,
            "timestamp": datetime.now().isoformat(),
            "direction": details.get("direction"),
            "expected": details.get("expected"),
            "speed_kmh": details.get("speed"),
        }
        if photo_path:
            # URL relativa — HA pode acessar via API
            payload["photo_url"] = f"/photos/{Path(photo_path).name}"
            payload["photo_file"] = str(photo_path)
        if clip_path:
            payload["clip_file"] = str(clip_path)
        
        self._publish("wrong-way", payload)
    
    def publish_speeding(self, track_id: int, details: dict, photo_path: str = None, clip_path: str = None):
        """Publica evento de excesso de velocidade."""
        payload = {
            "event": "speeding",
            "track_id": track_id,
            "timestamp": datetime.now().isoformat(),
            "speed_kmh": round(details.get("speed_kmh", 0), 1),
            "limit_kmh": details.get("limit_kmh"),
            "excess_kmh": round(details.get("speed_kmh", 0) - details.get("limit_kmh", 0), 1),
        }
        if photo_path:
            payload["photo_url"] = f"/photos/{Path(photo_path).name}"
            payload["photo_file"] = str(photo_path)
        if clip_path:
            payload["clip_file"] = str(clip_path)
        
        self._publish("speeding", payload)
    
    def publish_vehicle(self, track_id: int, class_name: str, speed_kmh: float | None):
        """Publica info de veículo detectado ( lightweight )."""
        payload = {
            "track_id": track_id,
            "class": class_name,
            "speed_kmh": round(speed_kmh, 1) if speed_kmh else None,
            "timestamp": datetime.now().isoformat(),
        }
        self._publish("vehicle", payload)
    
    def _setup_home_assistant_discovery(self):
        """Configura MQTT Auto-Discovery do Home Assistant."""
        if not self._connected:
            return
        
        device = {
            "identifiers": ["traffic-monitor"],
            "name": "Traffic Monitor",
            "manufacturer": "Traffic Monitor",
            "model": "CPU Camera Monitor",
        }
        
        sensors = [
            {
                "id": "status",
                "name": "Status",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ 'online' if value_json.online else 'offline' }}",
                "json_attributes_topic": f"{self.prefix}/status",
                "icon": "mdi:traffic-light",
            },
            {
                "id": "vehicles_total",
                "name": "Veículos Totais",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.vehicles_total }}",
                "icon": "mdi:car-multiple",
                "device_class": "sensor",
            },
            {
                "id": "wrong_way_total",
                "name": "Contramão Total",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.wrong_way_total }}",
                "icon": "mdi:car-alert",
            },
            {
                "id": "speeding_total",
                "name": "Excesso Velocidade Total",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.speeding_total }}",
                "icon": "mdi:speedometer-alert",
            },
            {
                "id": "last_speed",
                "name": "Última Velocidade",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.last_speed_kmh }}",
                "unit_of_measurement": "km/h",
                "icon": "mdi:speedometer",
            },
            {
                "id": "active_tracks",
                "name": "Veículos Ativos",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.active_tracks }}",
                "icon": "mdi:car",
            },
            {
                "id": "fps",
                "name": "FPS Processamento",
                "state_topic": f"{self.prefix}/status",
                "value_template": "{{ value_json.fps }}",
                "unit_of_measurement": "fps",
                "icon": "mdi:counter",
            },
            {
                "id": "wrong_way_event",
                "name": "Último Contramão",
                "state_topic": f"{self.prefix}/wrong-way",
                "value_template": "{{ value_json.direction }}",
                "json_attributes_topic": f"{self.prefix}/wrong-way",
                "icon": "mdi:car-alert",
            },
            {
                "id": "speeding_event",
                "name": "Último Excesso Velocidade",
                "state_topic": f"{self.prefix}/speeding",
                "value_template": "{{ value_json.speed_kmh }}",
                "unit_of_measurement": "km/h",
                "json_attributes_topic": f"{self.prefix}/speeding",
                "icon": "mdi:speedometer-alert",
            },
            # Camera virtual — mostra última foto
            {
                "id": "last_violation_photo",
                "name": "Última Infração (Foto)",
                "state_topic": f"{self.prefix}/wrong-way",
                "value_template": "{{ value_json.photo_url }}",
                "json_attributes_topic": f"{self.prefix}/speeding",
                "icon": "mdi:camera",
            },
        ]
        
        for sensor in sensors:
            sensor_id = sensor.pop("id")
            config_payload = {
                **sensor,
                "unique_id": f"traffic_monitor_{sensor_id}",
                "device": device,
                "expire_after": 120,  # Expira se parar de atualizar
            }
            
            # HA espera tópico: homeassistant/sensor/{node_id}/{object_id}/config
            topic = f"homeassistant/sensor/traffic_monitor/{sensor_id}/config"
            with self._lock:
                try:
                    self._client.publish(
                        topic,
                        payload=json.dumps(config_payload),
                        qos=1,
                        retain=True,
                    )
                except Exception as e:
                    logger.error(f"Erro discovery: {e}")
        
        logger.info("Home Assistant MQTT Discovery configurado")
    
    def disconnect(self):
        if self._client:
            self.publish_status({"online": False})
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT desconectado")
