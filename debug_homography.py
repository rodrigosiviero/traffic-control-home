#!/usr/bin/env python3
"""
Diagnóstico rápido da homografia.
Mostra o que a matriz H faz com os pontos de calibração
e com pontos de teste.
"""
import yaml
import numpy as np
import cv2
import sys

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    cal = config.get("calibration", {})
    method = cal.get("method", "linear")
    
    print(f"Método: {method}")
    
    if method != "homography" or "homography_matrix" not in cal:
        print("Não está usando homografia ou matriz não encontrada.")
        return
    
    H = np.array(cal["homography_matrix"], dtype=np.float64)
    print(f"\nMatriz H:\n{H}")
    
    # Testar pontos de calibração
    points = cal.get("points", [])
    print(f"\n=== Validando {len(points)} pontos de calibração ===")
    
    for i, pt in enumerate(points):
        px = np.array([[[pt["pixel"][0], pt["pixel"][1]]]], dtype=np.float32)
        real = cv2.perspectiveTransform(px, H)
        rx, ry = real[0][0][0], real[0][0][1]
        expected_x, expected_y = pt["real"]
        err_x = abs(rx - expected_x)
        err_y = abs(ry - expected_y)
        ok = "✓" if err_x < 0.5 and err_y < 0.5 else "✗"
        print(f"  P{i+1}: pixel {pt['pixel']} → ({rx:.2f}, {ry:.2f})m  "
              f"(esperado: ({expected_x}, {expected_y})m)  err: ({err_x:.2f}, {err_y:.2f}) {ok}")
    
    # Testar distância entre P1→P2 (deve ser 12m)
    if len(points) >= 2:
        p1 = np.array([[[points[0]["pixel"][0], points[0]["pixel"][1]]]], dtype=np.float32)
        p2 = np.array([[[points[1]["pixel"][0], points[1]["pixel"][1]]]], dtype=np.float32)
        r1 = cv2.perspectiveTransform(p1, H)[0][0]
        r2 = cv2.perspectiveTransform(p2, H)[0][0]
        dist = np.sqrt((r2[0]-r1[0])**2 + (r2[1]-r1[1])**2)
        print(f"\n  Distância P1→P2: {dist:.2f}m (esperado: 12.0m)")
    
    # Testar pontos de teste (rua)
    print(f"\n=== Testando pontos típicos de trânsito ===")
    
    test_points = [
        ("centro da rua (perto)", (700, 300)),
        ("centro da rua (longe)", (1000, 250)),
        ("beira inferior", (400, 400)),
        ("beira superior", (1100, 200)),
    ]
    
    pw = config["camera"].get("process_width", 1536)
    fw = config["camera"].get("process_width", 1536)
    # Se não tem process_height, calcula
    native_h = 576
    ratio = native_h / 1536
    ph = int(pw * ratio)
    
    print(f"  Resolução processamento: {pw}x{ph}")
    print(f"  (Calibração feita em 1536x576)")
    
    if pw != 1536:
        print(f"  ⚠ process_width={pw} != 1536 (resolução de calibração)")
        print(f"  Os pixels do tracker são escalados pro frame original (1536x576)")
        print(f"  Então a homografia deve funcionar... mas vamos verificar:")
    
    for name, (px_x, px_y) in test_points:
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, H)
        rx, ry = real[0][0][0], real[0][0][1]
        print(f"  {name}: pixel ({px_x},{px_y}) → ({rx:.2f}, {ry:.2f})m")
    
    # Simular velocidade
    print(f"\n=== Simulação de velocidade ===")
    # Carro andando 40 km/h = 11.1 m/s
    # Em 0.5s (5 frames a 10fps) percorre 5.6m
    # Na imagem, P1→P2 são 12m em ~(544,380)→(1148,297) = ~622 pixels
    # 5.6m ≈ 290 pixels no centro da imagem
    
    # Simular pontos de um carro a 40km/h
    sim_pixels = [(600, 320), (660, 315), (720, 310), (780, 305), (840, 300)]
    
    real_points = []
    for px_x, px_y in sim_pixels:
        pt = np.array([[[px_x, px_y]]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, H)
        real_points.append((real[0][0][0], real[0][0][1]))
    
    total_dist = 0
    for i in range(1, len(real_points)):
        dx = real_points[i][0] - real_points[i-1][0]
        dy = real_points[i][1] - real_points[i-1][1]
        d = np.sqrt(dx*dx + dy*dy)
        total_dist += d
        print(f"  Frame {i}: ({real_points[i][0]:.2f}, {real_points[i][1]:.2f})m  delta: {d:.2f}m")
    
    # FPS = 10 (padrão)
    for test_fps in [2.5, 5.0, 10.0]:
        elapsed = (len(sim_pixels) - 1) / test_fps
        speed_ms = total_dist / elapsed
        speed_kmh = speed_ms * 3.6
        print(f"\n  Total: {total_dist:.2f}m em {len(sim_pixels)-1} intervals")
        print(f"  Se FPS={test_fps}: {elapsed:.2f}s → {speed_kmh:.1f} km/h")
    
    print(f"\n  Velocidade REAL esperada: ~40 km/h")
    print(f"  Se o resultado tá ~240 km/h, o FPS tá errado (muito baixo)")


if __name__ == "__main__":
    main()
