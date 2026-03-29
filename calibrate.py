#!/usr/bin/env python3
"""
Ferramenta de calibração com HOMOGRAFIA para o Traffic Monitor v2.

Ajuda a determinar:
1. Pontos de referência para mapeamento pixel → mundo real (mínimo 4)
2. Direção correta do tráfego
3. Região de interesse (ROI)

A homografia corrige a perspectiva da câmera, permitindo calcular
velocidade precisa mesmo com câmera de lado/ângulo.

USO:
    python calibrate.py --rtsp rtsp://admin:senha@192.168.1.100:554/stream
    python calibrate.py --image frame.jpg

FLUXO:
    1. O programa abre a câmera ou imagem
    2. Pause apertando ESPAÇO (se câmera)
    3. Marque pelo menos 4 pontos com distâncias reais conhecidas
    4. Marque a direção do tráfego (2 pontos)
    5. Marque a ROI (4+ pontos)
    6. Aperte 's' para salvar

CONTROLES:
    ESPAÇO     - Pausar/continuar vídeo
    1          - Modo pontos de referência
    2          - Modo direção
    3          - Modo ROI
    c          - Limpar pontos do modo atual
    s          - Salvar config
    r          - Resetar tudo
    q          - Sair
    Mouse      - Clique para marcar pontos
"""

import argparse
import sys
import os
import yaml
import numpy as np
import cv2
from pathlib import Path

# Cores
COLOR_REF = (0, 255, 255)    # Amarelo - pontos de referência
COLOR_DIR = (255, 0, 255)    # Magenta - direção
COLOR_ROI = (0, 255, 0)      # Verde - ROI
COLOR_TEXT = (255, 255, 255)  # Branco
COLOR_BG = (0, 0, 0)         # Fundo


class HomographyCalibrator:
    def __init__(self, source, is_image=False):
        self.mode = "ref"  # ref, direction, roi
        self.ref_points = []      # [(pixel_x, pixel_y, real_x_m, real_y_m), ...]
        self.direction_points = [] # [pixel_pt1, pixel_pt2]
        self.roi_points = []       # [pixel_pt1, pixel_pt2, ...]
        
        self.paused = False
        self.current_input = ""  # Texto sendo digitado
        self.input_mode = False  # True quando digitando coordenadas reais
        self.input_axis = None   # "x" ou "y" - qual eixo está digitando
        self.pending_pixel = None # Pixel esperando coordenada real
        self.real_x = 0.0
        self.real_y = 0.0
        self.window_width = 1280
        
        # Abrir fonte
        if is_image:
            self.cap = None
            self.frame = cv2.imread(source)
            if self.frame is None:
                print(f"ERRO: Não abriu imagem {source}")
                sys.exit(1)
            h, w = self.frame.shape[:2]
            print(f"Imagem: {w}x{h}")
        else:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                print(f"ERRO: Não abriu {source}")
                sys.exit(1)
            # Ler um frame pra saber tamanho
            ret, self.frame = self.cap.read()
            if not ret:
                print("ERRO: Não leu frame")
                sys.exit(1)
            h, w = self.frame.shape[:2]
            print(f"Câmera: {w}x{h}")
        
        h, w = self.frame.shape[:2]
        self.native_w = w
        self.native_h = h
        
        # Escala pra janela
        self.scale = self.window_width / w
        self.display_h = int(h * self.scale)
        
        # Janela
        cv2.namedWindow("Calibracao v2 (Homografia)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibracao v2 (Homografia)", self.window_width, self.display_h)
        cv2.setMouseCallback("Calibracao v2 (Homografia)", self._on_mouse)
    
    def _to_display(self, frame):
        """Redimensiona frame pra janela."""
        return cv2.resize(frame, (self.window_width, self.display_h))
    
    def _to_native(self, display_x, display_y):
        """Converte coordenadas da janela pra nativas."""
        return int(display_x / self.scale), int(display_y / self.scale)
    
    def _on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        
        if self.input_mode:
            return  # Ignora clique enquanto digita
        
        nx, ny = self._to_native(x, y)
        
        if self.mode == "ref":
            # Primeiro clique: salvar pixel e pedir coordenada real X
            self.pending_pixel = (nx, ny)
            self.input_mode = True
            self.input_axis = "x"
            self.current_input = ""
            print(f"\n  Ponto pixel: ({nx}, {ny})")
            print(f"  Digite a coordenada real X (metros) e pressione ENTER:")
        
        elif self.mode == "direction":
            self.direction_points.append((nx, ny))
            print(f"  Direção ponto {len(self.direction_points)}: ({nx}, {ny})")
        
        elif self.mode == "roi":
            self.roi_points.append((nx, ny))
            print(f"  ROI ponto {len(self.roi_points)}: ({nx}, {ny})")
    
    def _handle_input(self, key):
        """Lida com entrada de texto para coordenadas reais."""
        if key == 13 or key == 10:  # Enter
            try:
                value = float(self.current_input)
            except ValueError:
                print("  Valor inválido! Digite um número.")
                self.current_input = ""
                return
            
            if self.input_axis == "x":
                self.real_x = value
                self.input_axis = "y"
                self.current_input = ""
                print(f"  X real = {value}m. Agora digite Y real (metros):")
            elif self.input_axis == "y":
                self.real_y = value
                px, py = self.pending_pixel
                self.ref_points.append((px, py, self.real_x, self.real_y))
                print(f"  ✓ Ponto #{len(self.ref_points)}: pixel=({px},{py}) → real=({self.real_x}m, {self.real_y}m)")
                self.input_mode = False
                self.pending_pixel = None
                self.current_input = ""
                
                # Se tem 4+ pontos, calcular homografia
                if len(self.ref_points) >= 4:
                    ok = self._test_homography()
                    if ok:
                        print(f"  ✅ Homografia calculada com {len(self.ref_points)} pontos")
        
        elif key == 27:  # ESC cancela input
            self.input_mode = False
            self.current_input = ""
            self.pending_pixel = None
            print("  Cancelado")
        
        elif key == 8 or key == 127:  # Backspace
            self.current_input = self.current_input[:-1]
        
        elif 48 <= key <= 57 or key == ord('.') or key == ord('-'):
            self.current_input += chr(key)
    
    def _test_homography(self):
        """Testa se a homografia é válida."""
        try:
            src = np.array([(p[0], p[1]) for p in self.ref_points], dtype=np.float32)
            dst = np.array([(p[2], p[3]) for p in self.ref_points], dtype=np.float32)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                print("  ❌ Homografia falhou - pontos podem ser colineares")
                return False
            
            # Testar reprojecção
            src_h = np.array([src], dtype=np.float32)
            projected = cv2.perspectiveTransform(src_h, H)
            error = np.mean(np.abs(projected[0] - dst))
            print(f"  Erro de reprojecção: {error:.3f}m")
            
            if error > 1.0:
                print(f"  ⚠️ Erro alto ({error:.1f}m) - revise os pontos")
                return False
            
            return True
        except Exception as e:
            print(f"  ❌ Erro: {e}")
            return False
    
    def _draw(self, frame):
        """Desenha overlays no frame."""
        display = frame.copy()
        
        # ROI
        if len(self.roi_points) >= 2:
            pts = np.array([(int(p[0]*self.scale), int(p[1]*self.scale)) for p in self.roi_points])
            cv2.polylines(display, [pts], len(self.roi_points) >= 3, COLOR_ROI, 2)
            for i, p in enumerate(self.roi_points):
                dp = (int(p[0]*self.scale), int(p[1]*self.scale))
                cv2.circle(display, dp, 5, COLOR_ROI, -1)
                cv2.putText(display, str(i+1), (dp[0]+8, dp[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROI, 1)
        
        # Pontos de referência
        for i, p in enumerate(self.ref_points):
            dp = (int(p[0]*self.scale), int(p[1]*self.scale))
            cv2.circle(display, dp, 8, COLOR_REF, -1)
            cv2.circle(display, dp, 8, COLOR_REF, 2)
            label = f"P{i+1} ({p[2]:.1f},{p[3]:.1f})m"
            cv2.putText(display, label, (dp[0]+12, dp[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_REF, 2)
        
        # Direção
        if len(self.direction_points) >= 1:
            for i, p in enumerate(self.direction_points):
                dp = (int(p[0]*self.scale), int(p[1]*self.scale))
                cv2.circle(display, dp, 8, COLOR_DIR, -1)
                cv2.putText(display, f"Dir {i+1}", (dp[0]+12, dp[1]+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIR, 2)
        if len(self.direction_points) == 2:
            dp1 = (int(self.direction_points[0][0]*self.scale), int(self.direction_points[0][1]*self.scale))
            dp2 = (int(self.direction_points[1][0]*self.scale), int(self.direction_points[1][1]*self.scale))
            cv2.arrowedLine(display, dp1, dp2, COLOR_DIR, 3)
        
        # Painel de info
        mode_names = {"ref": "REFERENCIA", "direction": "DIRECAO", "roi": "ROI"}
        mode_colors = {"ref": COLOR_REF, "direction": COLOR_DIR, "roi": COLOR_ROI}
        
        # Fundo
        panel_h = 160
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (self.window_width, panel_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        y = 25
        mode = mode_names.get(self.mode, "?")
        color = mode_colors.get(self.mode, COLOR_TEXT)
        cv2.putText(display, f"MODO: {mode}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 25
        cv2.putText(display, f"Pontos ref: {len(self.ref_points)}/min 4", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        y += 22
        cv2.putText(display, f"Direcao: {len(self.direction_points)}/2", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        y += 22
        cv2.putText(display, f"ROI: {len(self.roi_points)} pontos", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        
        # Input
        if self.input_mode:
            axis_name = "X" if self.input_axis == "x" else "Y"
            cv2.putText(display, f"Real {axis_name} (m): {self.current_input}_", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        
        # Controles (lado direito)
        cx = self.window_width - 350
        cv2.putText(display, "1=Ref  2=Dir  3=ROI", (cx, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(display, "SPACE=Pausa  c=Limpar  s=Salvar", (cx, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(display, "q=Sair  r=Reset", (cx, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        if len(self.ref_points) >= 4:
            cv2.putText(display, "✓ Homografia OK", (cx, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return display
    
    def _save_config(self):
        """Salva configuração em config.yaml."""
        if len(self.ref_points) < 4:
            print("❌ Precisa de pelo menos 4 pontos de referência!")
            return
        
        # Calcular homografia final
        src = np.array([(p[0], p[1]) for p in self.ref_points], dtype=np.float32)
        dst = np.array([(p[2], p[3]) for p in self.ref_points], dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if H is None:
            print("❌ Homografia inválida!")
            return
        
        # Determinar direção
        direction = "left_to_right"
        if len(self.direction_points) == 2:
            dx = self.direction_points[1][0] - self.direction_points[0][0]
            direction = "left_to_right" if dx > 0 else "right_to_left"
        
        # Carregar config existente ou usar defaults
        path = "config.yaml"
        if os.path.exists(path):
            with open(path, "r") as f:
                config = yaml.safe_load(f) or {}
            print(f"  Mesclando com config existente: {path}")
        else:
            config = {}
        
        # Atualizar seções de calibração (preserva o resto)
        config.setdefault("camera", {})["process_width"] = self.native_w
        config.setdefault("camera", {})["process_height"] = self.native_h
        config.setdefault("camera", {}).setdefault("skip_frames", 1)
        # Preserva rtsp_url existente
        config["camera"].setdefault("rtsp_url", "PREENCHA_AQUI")
        
        config["calibration"] = {
            "method": "homography",
            "points": [
                {"pixel": [p[0], p[1]], "real": [p[2], p[3]]}
                for p in self.ref_points
            ],
            "homography_matrix": H.tolist(),
        }
        
        config["direction"] = {
            "expected": direction,
            "tolerance": config.get("direction", {}).get("tolerance", 2.0),
        }
        
        config.setdefault("speed", {
            "limit_kmh": 40,
            "tolerance_kmh": 5,
            "min_track_frames": 3,
        })
        
        # Defaults que não podem faltar
        config.setdefault("detection", {
            "model_size": "n",
            "confidence": 0.3,
            "classes": [2, 3, 5, 7],
            "use_onnx": False,
        })
        config.setdefault("alerts", {
            "log_file": "logs/alerts.log",
            "save_clips": True,
            "clips_folder": "clips",
            "clip_duration_sec": 5,
            "terminal_beep": True,
            "cooldown_sec": 30,
        })
        config.setdefault("api", {"port": 8090, "status_interval": 30})
        config.setdefault("prometheus", {"enabled": True})
        config.setdefault("mqtt", {
            "enabled": False,
            "host": "localhost",
            "port": 1883,
            "username": "",
            "password": "",
            "topic_prefix": "traffic-monitor",
        })
        
        if self.roi_points:
            config["roi"] = {
                "polygon": [[p[0], p[1]] for p in self.roi_points]
            }
        
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Config salvo em {path}")
        print(f"   Pontos de referência: {len(self.ref_points)}")
        print(f"   Direção: {direction}")
        print(f"   ROI: {len(self.roi_points)} pontos")
        print(f"\n   MATRIZ DE HOMOGRAFIA (3x3):")
        for row in H:
            print(f"     [{', '.join(f'{v:.6f}' for v in row)}]")
    
    def run(self):
        print("\n" + "=" * 60)
        print("CALIBRAÇÃO V2 - HOMOGRAFIA")
        print("=" * 60)
        print("")
        print("COMO USAR:")
        print("  1. Pause o vídeo (ESPAÇO)")
        print("  2. Aperte '1' pra modo Referência")
        print("  3. Clique em pontos conhecidos na rua")
        print("  4. Pra cada ponto, digite X real (m) + ENTER, depois Y real (m) + ENTER")
        print("  5. Marque pelo menos 4 pontos espalhados pela área de visão")
        print("  6. Aperte '2' pra direção, '3' pra ROI")
        print("  7. 's' pra salvar")
        print("")
        print("DICAS PARA PONTOS:")
        print("  - Use marcas na rua, postes, bordas de calçada")
        print("  - Quanto mais espalhados, melhor a correção de perspectiva")
        print("  - Meça distâncias reais com trena/google maps")
        print("  - Mínimo 4 pontos, ideal 6-8")
        print("")
        
        while True:
            # Capturar frame
            if self.cap and not self.paused:
                ret, new_frame = self.cap.read()
                if ret:
                    self.frame = new_frame
            
            # Desenhar
            display = self._draw(self.frame)
            cv2.imshow("Calibracao v2 (Homografia)", display)
            
            key = cv2.waitKey(30 if not self.paused else 100) & 0xFF
            
            # Input mode
            if self.input_mode:
                self._handle_input(key)
                continue
            
            if key == ord("q"):
                break
            elif key == ord(" "):
                self.paused = not self.paused
                print("PAUSADO" if self.paused else "PLAY")
            elif key == ord("1"):
                self.mode = "ref"
                print("Modo: REFERÊNCIA - clique em pontos conhecidos")
            elif key == ord("2"):
                self.mode = "direction"
                print("Modo: DIREÇÃO - clique 2 pontos (origem → destino)")
            elif key == ord("3"):
                self.mode = "roi"
                print("Modo: ROI - clique pontos do polígono")
            elif key == ord("c"):
                if self.mode == "ref":
                    self.ref_points.clear()
                    print("Referências limpas")
                elif self.mode == "direction":
                    self.direction_points.clear()
                    print("Direção limpa")
                elif self.mode == "roi":
                    self.roi_points.clear()
                    print("ROI limpa")
            elif key == ord("s"):
                self._save_config()
            elif key == ord("r"):
                self.ref_points.clear()
                self.direction_points.clear()
                self.roi_points.clear()
                print("Tudo resetado")
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Calibração v2 - Homografia")
    parser.add_argument("--rtsp", help="URL RTSP da câmera")
    parser.add_argument("--image", help="Imagem/frame para calibrar")
    args = parser.parse_args()
    
    if args.image:
        cal = HomographyCalibrator(args.image, is_image=True)
    elif args.rtsp:
        cal = HomographyCalibrator(args.rtsp)
    else:
        print("Use --rtsp <url> ou --image <arquivo>")
        sys.exit(1)
    
    cal.run()


if __name__ == "__main__":
    main()
