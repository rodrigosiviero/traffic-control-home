import logging
import numpy as np

logger = logging.getLogger("traffic-monitor.tracker")


class VehicleTracker:
    """
    Rastreamento de veículos entre frames.
    
    Usa IoU + distância para associar detecções a tracks existentes.
    """
    
    def __init__(self, config: dict):
        self.max_age = config.get("max_track_age", 90)
        self.min_hits = config.get("min_track_frames", 3)
        self.iou_threshold = config.get("iou_threshold", 0.1)
        self.max_distance = config.get("max_track_distance", 400)  # pixels
        
        self.tracks = {}
        self._next_id = 1
    
    def update(self, detections: list, frame_num: int) -> dict:
        """
        Atualiza tracks com novas detecções.
        
        Args:
            detections: [[x1, y1, x2, y2, conf, class_id], ...]
            frame_num: número do frame atual
        
        Returns:
            dict de tracks ativos: {track_id: track_data}
        """
        # Sem detecções - envelhecer tracks
        if not detections:
            to_remove = []
            for tid, track in self.tracks.items():
                track["age"] += 1
                if track["age"] > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return {}
        
        # Sem tracks - criar todos
        if not self.tracks:
            for det in detections:
                self._create_track(det, frame_num)
            return {}
        
        # Matchear usando IoU + distância
        track_ids = list(self.tracks.keys())
        det_boxes = [d[:4] for d in detections]
        
        # Calcular custo: menor = melhor match
        # Custo = 1 - IoU (para matching húngaro/greedy)
        n_tracks = len(track_ids)
        n_dets = len(detections)
        
        cost_matrix = np.full((n_tracks, n_dets), 1e6)
        
        for i, tid in enumerate(track_ids):
            last_box = self.tracks[tid]["last_detection"][:4]
            last_center = self._box_center(self.tracks[tid]["last_detection"])
            
            for j in range(n_dets):
                det_center = self._box_center(detections[j])
                
                # 1. Distância entre centros
                dist = np.sqrt(
                    (last_center[0] - det_center[0]) ** 2 +
                    (last_center[1] - det_center[1]) ** 2
                )
                
                # Se muito longe, pular
                if dist > self.max_distance:
                    continue
                
                # 2. IoU
                iou = self._compute_iou(last_box, det_boxes[j])
                
                # Custo combinado: priorizar IoU alto + distância baixa
                if iou > self.iou_threshold:
                    cost_matrix[i, j] = -iou  # Negativo: menor custo = melhor
                elif dist < self.max_distance * 0.7:
                    # Fallback por distância (se perto o suficiente)
                    cost_matrix[i, j] = 1.0 + (dist / self.max_distance)
        
        # Greedy matching
        matched_tracks = set()
        matched_dets = set()
        
        # Ordenar por menor custo
        flat_indices = np.argsort(cost_matrix, axis=None)
        for idx in flat_indices:
            i = int(idx // n_dets)
            j = int(idx % n_dets)
            
            if cost_matrix[i, j] >= 1e6:
                break  # Resto é inválido
            
            if i in matched_tracks or j in matched_dets:
                continue
            
            tid = track_ids[i]
            center = self._box_center(detections[j])
            self.tracks[tid]["history"].append(center)
            self.tracks[tid]["last_detection"] = detections[j]
            self.tracks[tid]["age"] = 0
            self.tracks[tid]["last_frame"] = frame_num
            matched_tracks.add(i)
            matched_dets.add(j)
        
        # Novos tracks
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(detections[j], frame_num)
        
        # Envelhecer não-matcheados e remover velhos
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]
        
        # Retornar tracks com histórico suficiente
        active = {}
        for tid, track in self.tracks.items():
            if len(track["history"]) >= 2:
                active[tid] = track
        
        return active
    
    def _create_track(self, detection, frame_num):
        center = self._box_center(detection)
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "history": [center],
            "last_detection": detection,
            "age": 0,
            "last_frame": frame_num,
            "start_frame": frame_num,
        }
    
    @staticmethod
    def _box_center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    @staticmethod
    def _compute_iou(box_a, box_b):
        """IoU entre duas boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        
        return intersection / union
