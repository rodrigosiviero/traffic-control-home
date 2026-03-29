#!/usr/bin/env python3
"""
Teste de detecção com arquivo de vídeo.

Uso:
    python test_video.py --video video.mp4
    python test_video.py --video video.mp4 --loop --confidence 0.2 --width 1280
"""
import argparse
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    parser = argparse.ArgumentParser(description="Teste de detecção com vídeo")
    parser.add_argument("--video", required=True, help="Caminho do vídeo")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confiança mínima")
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 3, 5, 7])
    parser.add_argument("--loop", action="store_true", help="Repetir em loop")
    parser.add_argument("--width", type=int, default=1280, help="Largura da janela")
    parser.add_argument("--save", default="", help="Salvar vídeo anotado")
    args = parser.parse_args()

    print(f"Modelo: {args.model}")
    model = YOLO(args.model)
    
    print(f"Vídeo: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERRO: Não abriu {args.video}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Janela com tamanho fixo (redimensionável pelo usuário também)
    cv2.namedWindow("Detection Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detection Test", args.width, int(args.width * h / w))
    
    print(f"Resolução: {w}x{h} | FPS: {video_fps:.0f} | Frames: {total_frames}")
    print(f"Classes: {[COCO_CLASSES.get(c, str(c)) for c in args.classes]}")
    print(f"Confiança: {args.confidence} | Janela: {args.width}px largura")
    print("q=sair | espaço=pausar | +=mais conf | -=menos conf")
    print("")
    
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, video_fps, (w, h))
        print(f"Salvando: {args.save}")
    
    confidence = args.confidence
    paused = False
    frame_num = 0
    total_detections = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("--- Loop ---")
                    continue
                else:
                    print(f"Fim. Detecções: {total_detections}")
                    break
            
            if paused:
                # Mostrar frame sem processar
                disp = cv2.resize(frame, (args.width, int(h * args.width / w)))
                cv2.putText(disp, "PAUSADO (espaco p/ continuar)", (10, disp.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Detection Test", disp)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    paused = False
                elif key in (ord("+"), ord("=")):
                    confidence = min(1.0, confidence + 0.05)
                    print(f"Conf: {confidence:.2f}")
                elif key == ord("-"):
                    confidence = max(0.05, confidence - 0.05)
                    print(f"Conf: {confidence:.2f}")
                continue
            
            frame_num += 1
            
            # Detectar no frame ORIGINAL (resolução cheia)
            t0 = time.time()
            results = model.predict(frame, conf=confidence, classes=args.classes,
                                   verbose=False, device="cpu")
            dt = time.time() - t0
            
            # Desenhar boxes no frame original
            det_count = 0
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                det_count = len(boxes)
                total_detections += det_count
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    cls_name = COCO_CLASSES.get(cls_id, str(cls_id))
                    
                    colors = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 255, 255), 7: (0, 165, 255)}
                    color = colors.get(cls_id, (255, 255, 255))
                    label = f"{cls_name} {conf:.0%}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Info no topo
            fps_real = 1.0 / dt if dt > 0 else 0
            cv2.putText(frame,
                       f"Frame {frame_num}/{total_frames} | "
                       f"FPS: {fps_real:.0f} | "
                       f"{dt*1000:.0f}ms | "
                       f"Det: {det_count} | "
                       f"Conf: {confidence:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Redimensionar pra janela
            display = cv2.resize(frame, (args.width, int(h * args.width / w)))
            cv2.imshow("Detection Test", display)
            
            if writer:
                writer.write(frame)
            
            key = cv2.waitKey(max(1, int(1000 / video_fps) - int(dt * 1000))) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = True
            elif key in (ord("+"), ord("=")):
                confidence = min(1.0, confidence + 0.05)
                print(f"Conf: {confidence:.2f}")
            elif key == ord("-"):
                confidence = max(0.05, confidence - 0.05)
                print(f"Conf: {confidence:.2f}")
    
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nResumo: {frame_num} frames, {total_detections} detecções")


if __name__ == "__main__":
    main()
