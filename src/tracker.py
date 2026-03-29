import logging
from collections import deque
import numpy as np

logger = logging.getLogger("traffic-monitor.tracker")

HISTORY_MAXLEN = 30  # Pontos de histórico por track (descarta antigos automaticamente)


class VehicleTracker:
    """
    Rastreamento de veículos entre frames.
    
    Usa IoU + distância (vetorizado com numpy) para associar
    detecções a tracks existentes.
    """

    def __init__(self, config: dict):
        self.max_age = config.get("max_track_age", 90)
        self.min_hits = config.get("min_track_frames", 3)
        self.iou_threshold = config.get("iou_threshold", 0.1)
        self.max_distance = config.get("max_track_distance", 400)

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
        # Sem detecções — envelhecer todos
        if not detections:
            self._age_all_tracks()
            return {}

        # Sem tracks — criar todos
        if not self.tracks:
            for det in detections:
                self._create_track(det, frame_num)
            return {}

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(detections)

        # Matriz de custo vetorizada
        cost_matrix = self._build_cost_matrix(track_ids, detections)

        # Greedy matching
        matched_tracks = set()
        matched_dets = set()

        flat_sorted = np.argsort(cost_matrix, axis=None)
        for idx in flat_sorted:
            i, j = divmod(int(idx), n_dets)
            if cost_matrix[i, j] >= 1e6:
                break
            if i in matched_tracks or j in matched_dets:
                continue

            tid = track_ids[i]
            det = detections[j]
            center = ((det[0] + det[2]) / 2, (det[1] + det[3]) / 2)

            t = self.tracks[tid]
            t["history"].append(center)
            t["last_detection"] = det
            t["age"] = 0
            t["last_frame"] = frame_num
            matched_tracks.add(i)
            matched_dets.add(j)

        # Novos tracks para detecções não matcheadas
        for j in range(n_dets):
            if j not in matched_dets:
                self._create_track(detections[j], frame_num)

        # Envelhecer não-matcheados
        self._age_unmatched(matched_tracks, track_ids)

        # Retornar tracks com histórico suficiente
        return {
            tid: t for tid, t in self.tracks.items()
            if len(t["history"]) >= 2
        }

    def _build_cost_matrix(self, track_ids, detections):
        """Calcula matriz de custo inteira de uma vez (vetorizado)."""
        n_t = len(track_ids)
        n_d = len(detections)

        # Boxes: (T, 4) e (D, 4)
        tb = np.array([self.tracks[tid]["last_detection"][:4] for tid in track_ids])
        db = np.array([d[:4] for d in detections])

        # Centros: (T, 2) e (D, 2)
        tc = np.column_stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2])
        dc = np.column_stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2])

        # Distâncias: broadcasting (T, 1, 2) - (1, D, 2) → (T, D)
        diff = tc[:, None, :] - dc[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))

        # IoU vetorizado: broadcasting (T, 1, 4) e (1, D, 4)
        ix1 = np.maximum(tb[:, None, 0], db[None, :, 0])
        iy1 = np.maximum(tb[:, None, 1], db[None, :, 1])
        ix2 = np.minimum(tb[:, None, 2], db[None, :, 2])
        iy2 = np.minimum(tb[:, None, 3], db[None, :, 3])

        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        area_t = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
        area_d = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1])
        union = area_t[:, None] + area_d[None, :] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Montar cost matrix
        too_far = dist > self.max_distance
        good_iou = (iou > self.iou_threshold) & ~too_far
        fallback = ~good_iou & ~too_far & (dist < self.max_distance * 0.7)

        cost = np.full((n_t, n_d), 1e6)
        cost = np.where(good_iou, -iou, cost)
        cost = np.where(fallback & (cost >= 1e6), 1.0 + dist / self.max_distance, cost)

        return cost

    def _age_all_tracks(self):
        """Envelhece todos os tracks (sem detecções no frame)."""
        to_remove = []
        for tid, t in self.tracks.items():
            t["age"] += 1
            if t["age"] > self.max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def _age_unmatched(self, matched_indices, track_ids):
        """Envelhece tracks que não foram matcheados."""
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_indices:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def _create_track(self, detection, frame_num):
        """Cria novo track a partir de uma detecção."""
        center = ((detection[0] + detection[2]) / 2,
                  (detection[1] + detection[3]) / 2)
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = {
            "id": tid,
            "history": deque([center], maxlen=HISTORY_MAXLEN),
            "last_detection": detection,
            "age": 0,
            "last_frame": frame_num,
            "start_frame": frame_num,
        }
