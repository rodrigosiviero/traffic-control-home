#!/usr/bin/env python3
"""
Profiler de CPU — roda 100 frames e mostra onde tá o tempo.

USO: python profile_cpu.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import time
import cv2
import yaml
import numpy as np
from collections import defaultdict

print("=== Traffic Monitor CPU Profiler ===\n")

# Carregar config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

rtsp_url = config["camera"]["rtsp_url"]
skip_frames = config["camera"].get("skip_frames", 1)
process_width = config["camera"].get("process_width", 1536)

print(f"RTSP: {rtsp_url[:30]}...")
print(f"skip_frames: {skip_frames}")
print(f"process_width: {process_width}")
print()

# Conectar câmera
print("Conectando câmera...")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("ERRO: não conectou")
    exit(1)
print("OK\n")

# Carregar modelo
print("Carregando YOLO...")
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.predict(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)  # warmup
print("OK\n")

# Medir
timings = defaultdict(list)
total_frames = 100
frame_count = 0
processed = 0

print(f"Processando {total_frames} frames...\n")

while frame_count < total_frames:
    t_read_start = time.perf_counter()
    ret, frame = cap.read()
    t_read = time.perf_counter() - t_read_start
    
    if not ret:
        print("Frame perdido")
        break
    
    frame_count += 1
    timings["read"].append(t_read)
    
    # Skip
    if frame_count % (skip_frames + 1) != 0:
        timings["skip"].append(time.perf_counter() - t_read_start)
        continue
    
    processed += 1
    
    # Resize
    t0 = time.perf_counter()
    h, w = frame.shape[:2]
    if process_width != w:
        ratio = process_width / w
        ph = int(h * ratio)
        small = cv2.resize(frame, (process_width, ph))
    else:
        small = frame
    timings["resize"].append(time.perf_counter() - t0)
    
    # YOLO
    t0 = time.perf_counter()
    results = model.predict(small, verbose=False, conf=0.3, classes=[2, 3, 5, 7])
    t_yolo = time.perf_counter() - t0
    timings["yolo"].append(t_yolo)
    
    # Extração de detecções
    t0 = time.perf_counter()
    dets = []
    if results and len(results[0].boxes):
        boxes = results[0].boxes
        data = boxes.data.cpu().numpy()
        dets = data.tolist()
    timings["extract"].append(time.perf_counter() - t0)
    
    # Simular tracker (10 tracks, 10 dets)
    t0 = time.perf_counter()
    if len(dets) > 0:
        tb = np.random.rand(10, 4) * 500
        db = np.array([d[:4] for d in dets[:10]])
        if len(db) > 0:
            tc = np.column_stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2])
            dc = np.column_stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2])
            diff = tc[:, None, :] - dc[None, :, :]
            dist = np.sqrt((diff ** 2).sum(axis=2))
    timings["tracker"].append(time.perf_counter() - t0)
    
    # Simular frame.copy() (alert buffer)
    t0 = time.perf_counter()
    _ = frame.copy()
    timings["frame_copy"].append(time.perf_counter() - t0)

cap.release()

# Resultados
print(f"Frames lidos: {frame_count}")
print(f"Frames processados: {processed}")
print(f"Ratio: {processed/frame_count*100:.0f}% (skip_frames={skip_frames})")
print()

print(f"{'Operação':<20} {'Média (ms)':<12} {'Total (ms)':<12} {'% do total':<12}")
print("-" * 56)

all_times = {}
for name, times_list in timings.items():
    total = sum(times_list) * 1000
    avg = (total / len(times_list)) if times_list else 0
    all_times[name] = total

grand_total = sum(all_times.values())

for name, total in sorted(all_times.items(), key=lambda x: -x[1]):
    avg = total / len(timings[name]) if timings[name] else 0
    pct = total / grand_total * 100
    print(f"{name:<20} {avg:<12.1f} {total:<12.0f} {pct:<12.1f}")

print("-" * 56)
print(f"{'TOTAL':<20} {'':12} {grand_total:<12.0f} {'100.0':12}")

# FPS estimado
wall_time = sum(timings["read"]) + sum(timings.get("yolo", [0]))
if wall_time > 0:
    fps = processed / wall_time
    print(f"\nFPS de processamento: {fps:.1f}")
    print(f"CPU estimado (1 core): {min(100, 200/fps * 50):.0f}%")

# Diagnóstico
print("\n=== DIAGNÓSTICO ===")
yolo_pct = all_times.get("yolo", 0) / grand_total * 100
read_pct = all_times.get("read", 0) / grand_total * 100
copy_pct = all_times.get("frame_copy", 0) / grand_total * 100

if yolo_pct > 60:
    print(f"⚠ YOLO é {yolo_pct:.0f}% do custo. Opções:")
    print(f"  - Aumentar skip_frames (atual: {skip_frames})")
    print(f"  - Diminuir process_width (atual: {process_width})")
    print(f"  - Usar intervalo maior no YOLO (imgsz=320)")

if read_pct > 20:
    print(f"⚠ Leitura de frames é {read_pct:.0f}% — câmera mandando frames demais")

if copy_pct > 5:
    print(f"⚠ frame.copy() é {copy_pct:.0f}% — considerar buffer menor")
