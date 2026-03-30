#!/usr/bin/env python3
"""
Exporta modelo YOLO para OpenVINO e testa na Intel GPU.

USO:
    python export_openvino.py          # Exporta yolov8n
    python export_openvino.py --size s # Exporta yolov8s (mais preciso, mais lento)
    python export_openvino.py --test   # Exporta + testa velocidade
"""
import argparse
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="n", help="Modelo: n (nano), s (small), m (medium)")
    parser.add_argument("--test", action="store_true", help="Testar velocidade após exportar")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolução interna")
    args = parser.parse_args()

    from ultralytics import YOLO

    model_name = f"yolov8{args.size}"
    print(f"Carregando {model_name}.pt...")
    model = YOLO(f"{model_name}.pt")

    print(f"Exportando para OpenVINO (imgsz={args.imgsz})...")
    model.export(format="openvino", imgsz=args.imgsz, half=False)
    print("✓ Exportado!")

    if args.test:
        print("\nTestando velocidade...")
        
        # Verificar devices
        try:
            from openvino.runtime import Core
            core = Core()
            print(f"Devices: {core.available_devices}")
            for d in core.available_devices:
                try:
                    name = core.get_property(d, "FULL_DEVICE_NAME")
                    print(f"  {d}: {name}")
                except:
                    print(f"  {d}: (disponível)")
        except ImportError:
            print("OpenVINO não instalado — teste só em PyTorch")

        # Testar CPU
        print(f"\n--- PyTorch CPU ---")
        pt_model = YOLO(f"{model_name}.pt")
        dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        
        # Warmup
        pt_model.predict(dummy, verbose=False, imgsz=args.imgsz)
        
        times = []
        for i in range(20):
            t0 = time.perf_counter()
            pt_model.predict(dummy, verbose=False, imgsz=args.imgsz)
            times.append(time.perf_counter() - t0)
        avg_pt = np.mean(times) * 1000
        print(f"  Média: {avg_pt:.1f}ms ({1000/avg_pt:.1f} fps)")

        # Testar OpenVINO
        print(f"\n--- OpenVINO ---")
        ov_model = YOLO(f"{model_name}_openvino_model/", task="detect")
        ov_model.predict(dummy, verbose=False, imgsz=args.imgsz)  # warmup
        
        times = []
        for i in range(20):
            t0 = time.perf_counter()
            ov_model.predict(dummy, verbose=False, imgsz=args.imgsz)
            times.append(time.perf_counter() - t0)
        avg_ov = np.mean(times) * 1000
        print(f"  Média: {avg_ov:.1f}ms ({1000/avg_ov:.1f} fps)")
        print(f"  Speedup: {avg_pt/avg_ov:.1f}x")


if __name__ == "__main__":
    main()
