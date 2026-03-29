"""Utilitários gerais."""
import numpy as np


def box_center(box) -> tuple:
    """Retorna centro de uma bounding box [x1,y1,x2,y2]."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_distance(p1, p2) -> float:
    """Distância euclidiana entre dois pontos."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def draw_grid(frame, spacing=50, color=(50, 50, 50), thickness=1):
    """Desenha grid no frame (útil para calibração visual)."""
    h, w = frame.shape[:2]
    for x in range(0, w, spacing):
        cv2_line(frame, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv2_line(frame, (0, y), (w, y), color, thickness)


def format_speed(speed_kmh: float | None) -> str:
    if speed_kmh is None:
        return "-- km/h"
    return f"{speed_kmh:.0f} km/h"
