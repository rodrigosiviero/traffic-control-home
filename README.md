# Traffic Monitor - Detecção de Velocidade e Direção

Monitoramento de veículos via câmera RTSP com:
- **Detecção de contramão**
- **Estimativa de velocidade**
- **Gravação de clips** (buffer retrospectivo + continuação)
- **Prometheus metrics** (dashboard Grafana)
- **MQTT** (Home Assistant)
- **API HTTP** (serve fotos e clips)

Funciona **100% em CPU**.

## Quick Start

```bash
# 1. Configurar
cp config.example.yaml config.yaml
vim config.yaml  # Ajustar RTSP, calibração, MQTT, etc.

# 2. Calibrar (requer display)
python calibrate.py --rtsp rtsp://admin:senha@IP:554/stream

# 3. Rodar local
pip install -r requirements.txt
python main.py --debug

# 4. Ou rodar em Docker
docker compose up -d
```

## Docker

```bash
# Build e rodar
docker compose up -d

# Logs
docker compose logs -f

# Parar
docker compose down

# Rebuild
docker compose up -d --build
```

### Portas

| Porta | Uso |
|-------|-----|
| 8090 | API HTTP (metrics + fotos + clips) |

### Volumes

| Path | Uso |
|------|-----|
| `./config.yaml` | Configuração (read-only) |
| `./data/clips/` | Vídeos e screenshots das infrações |
| `./data/logs/` | Logs do sistema |

## Integrações

### Prometheus

Adicionar ao `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'traffic-monitor'
    scrape_interval: 10s
    static_configs:
      - targets: ['IP:8090']
```

Importar dashboard: `grafana/traffic-monitor-dashboard.json`

### MQTT / Home Assistant

Habilitar no `config.yaml`:
```yaml
mqtt:
  enabled: true
  host: "192.168.15.97"
  port: 1883
```

Sensores aparecem automaticamente via MQTT Auto-Discovery.
Ver `homeassistant/README.md` para automações e cards.

## API Endpoints

| Endpoint | Descrição |
|----------|-----------|
| `GET /metrics` | Métricas Prometheus |
| `GET /status` | Status JSON |
| `GET /health` | Health check |
| `GET /photos/{name}` | Foto de infração |
| `GET /clips/{name}` | Clip de vídeo |

## Estrutura

```
traffic-monitor/
├── main.py              # Entrada principal
├── calibrate.py         # Ferramenta de calibração
├── config.example.yaml  # Configuração exemplo
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── src/
│   ├── camera.py        # Captura RTSP
│   ├── detector.py      # YOLO (CPU/ONNX)
│   ├── tracker.py       # Rastreamento IoU
│   ├── speed.py         # Estimativa velocidade
│   ├── direction.py     # Detecção direção
│   ├── alert.py         # Alertas + clips + MQTT + metrics
│   ├── metrics.py       # Prometheus metrics
│   ├── mqtt_client.py   # MQTT publisher
│   └── api.py           # Servidor HTTP
├── grafana/
│   └── traffic-monitor-dashboard.json
├── homeassistant/
│   └── README.md
└── data/                # Dados persistentes (gitignored)
    ├── clips/
    └── logs/
```
