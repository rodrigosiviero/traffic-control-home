# Home Assistant - Configuração

O Traffic Monitor se integra ao HA via **MQTT Auto-Discovery**.
Se o MQTT está habilitado no HA, os sensores aparecem automaticamente!

## 1. Habilitar MQTT no config.yaml do HA

```yaml
mqtt:
  broker: core-mosquitto
  # ou o IP do seu broker MQTT
```

## 2. Habilitar MQTT no config.yaml do Traffic Monitor

```yaml
mqtt:
  enabled: true
  host: "192.168.15.97"   # IP do broker MQTT
  port: 1883
  username: ""             # Se usar autenticação
  password: ""
```

## 3. Sensores que aparecem automaticamente

| Sensor | O que mostra |
|--------|-------------|
| `sensor.traffic_monitor_status` | Online/Offline + atributos detalhados |
| `sensor.traffic_monitor_veiculos_totais` | Total de veículos detectados |
| `sensor.traffic_monitor_contra_mao_total` | Total contramão |
| `sensor.traffic_monitor_excesso_velocidade_total` | Total acima do limite |
| `sensor.traffic_monitor_ultima_velocidade` | Velocidade do último veículo |
| `sensor.traffic_monitor_veiculos_ativos` | Veículos sendo rastreados |
| `sensor.traffic_monitor_fps` | FPS do processamento |
| `sensor.traffic_monitor_ultimo_contra_mao` | Detalhes último contramão |
| `sensor.traffic_monitor_ultimo_excesso_velocidade` | Detalhes último excesso |

## 4. Automações sugeridas

### Notificação de Contramão (adicionar ao automations.yaml)

```yaml
- alias: "Traffic Monitor - Contramão Detectado"
  trigger:
    - trigger: mqtt
      topic: "traffic-monitor/wrong-way"
  action:
    - action: notify.mobile_app_seu_celular
      data:
        title: "⚠️ Contramão Detectada!"
        message: >
          Veículo na contramão!
          Direção: {{ trigger.payload_json.direction }}
          Velocidade: {{ trigger.payload_json.speed_kmh | default('N/A') }} km/h
          Hora: {{ now().strftime('%H:%M:%S') }}
        data:
          push:
            sound:
              name: default
              critical: 1
          image: >
            http://IP_DO_SERVIDOR:8090{{ trigger.payload_json.photo_url | default('') }}
```

### Notificação de Excesso de Velocidade

```yaml
- alias: "Traffic Monitor - Excesso Velocidade"
  trigger:
    - trigger: mqtt
      topic: "traffic-monitor/speeding"
  action:
    - action: notify.mobile_app_seu_celular
      data:
        title: "🚀 Excesso de Velocidade!"
        message: >
          {{ trigger.payload_json.speed_kmh }} km/h 
          (limite: {{ trigger.payload_json.limit_kmh }} km/h)
          Excesso: {{ trigger.payload_json.excess_kmh }} km/h
        data:
          image: >
            http://IP_DO_SERVIDOR:8090{{ trigger.payload_json.photo_url | default('') }}
```

### Card no Dashboard (adicionar ao dashboard YAML)

```yaml
type: vertical-stack
cards:
  - type: glance
    title: Traffic Monitor
    entities:
      - sensor.traffic_monitor_status
      - sensor.traffic_monitor_veiculos_totais
      - sensor.traffic_monitor_contra_mao_total
      - sensor.traffic_monitor_excesso_velocidade_total
      - sensor.traffic_monitor_ultima_velocidade
  
  - type: conditional
    conditions:
      - entity: sensor.traffic_monitor_ultimo_contra_mao
        state_not: "unknown"
    card:
      type: picture
      image: >
        {% set photo = state_attr('sensor.traffic_monitor_ultimo_contra_mao', 'photo_url') %}
        http://IP_DO_SERVIDOR:8090{{ photo | default('') }}
      title: "Última Infração"
```

## 5. Camera entity (webcam picture)

Para mostrar a foto da última infração como uma "camera" no HA:

```yaml
# configuration.yaml
camera:
  - platform: generic
    name: Traffic Monitor - Última Infração
    still_image_url: >
      {% set last = state_attr('sensor.traffic_monitor_ultimo_contra_mao', 'photo_url') %}
      http://IP_DO_SERVIDOR:8090{{ last | default('/photos/') }}
    content_type: image/jpeg
    verify_ssl: false
```
