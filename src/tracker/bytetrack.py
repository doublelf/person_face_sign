"""轻量 IoU 跟踪器 (ByteTrack-style, 无 Kalman 滤波版本)。

适合本项目的简化版本: 行人跟踪 + 采脸节流。
    - 每个 track 有 age (从最后一次 hit 起的帧数) 和 hit_count
    - 关联: 优先按 IoU 匹配,新检测创建新 track
    - 死亡: age > max_age 时删除

不做:
    - 两阶段高低分关联 (那是完整 ByteTrack)
    - 卡尔曼滤波预测 (CPU 节省)

因为本项目最终目标不是"完美跟踪",而是"跨帧同人采脸节流",所以简化即可。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    age: int = 0
    hits: int = 1
    last_seen: int = 0
    sample_count: int = 0  # 该 track 已采脸次数
    next_sample_frame: int = 0  # 下次可采脸的全局帧号


class IoUTracker:
    """IoU-based tracker (简化版 ByteTrack)。"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.30,
        sample_interval_frames: int = 90,  # 25 FPS 时约 3.6 秒采一次脸
        max_samples_per_track: int = 3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.sample_interval_frames = sample_interval_frames
        self.max_samples_per_track = max_samples_per_track
        self._tracks: List[Track] = []
        self._next_id = 1
        self._frame_id = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_id = 0

    @staticmethod
    def _iou(b1: Tuple[float, float, float, float],
             b2: Tuple[float, float, float, float]) -> float:
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-9)

    def _greedy_match(
        self,
        detections: List[Tuple[float, float, float, float, float]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian-lite greedy match: 按 IoU 降序贪心匹配。

        Returns:
            matched: (det_idx, track_idx)
            unmatched_det: list of det_idx
            unmatched_track: list of track_idx
        """
        n_det = len(detections)
        n_trk = len(self._tracks)
        if n_det == 0:
            return [], [], list(range(n_trk))
        if n_trk == 0:
            return [], list(range(n_det)), []

        # 计算 IoU 矩阵
        iou_mat = np.zeros((n_det, n_trk), dtype=np.float32)
        for di, (x1, y1, x2, y2, _sc) in enumerate(detections):
            for ti, t in enumerate(self._tracks):
                iou_mat[di, ti] = self._iou((x1, y1, x2, y2), t.bbox)

        matched = []
        used_det = set()
        used_trk = set()
        # 贪心: 按 IoU 降序匹配
        order = np.argsort(-iou_mat.flatten())
        for idx in order:
            di, ti = int(idx / n_trk), int(idx % n_trk)
            if iou_mat[di, ti] < self.iou_threshold:
                break
            if di in used_det or ti in used_trk:
                continue
            matched.append((di, ti))
            used_det.add(di)
            used_trk.add(ti)
        unmatched_det = [i for i in range(n_det) if i not in used_det]
        unmatched_trk = [i for i in range(n_trk) if i not in used_trk]
        return matched, unmatched_det, unmatched_trk

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]],
    ) -> List[Track]:
        """推进一帧。

        Args:
            detections: [(x1, y1, x2, y2, score), ...] 当帧所有行人检测

        Returns:
            所有 active tracks (含刚创建的), caller 据此判断谁该采脸
        """
        self._frame_id += 1
        matched, unmatched_det, unmatched_trk = self._greedy_match(detections)

        # 更新匹配上的 tracks
        for di, ti in matched:
            x1, y1, x2, y2, sc = detections[di]
            t = self._tracks[ti]
            t.bbox = (x1, y1, x2, y2)
            t.score = sc
            t.age = 0
            t.hits += 1
            t.last_seen = self._frame_id

        # 未匹配的 tracks: age++
        for ti in unmatched_trk:
            self._tracks[ti].age += 1

        # 未匹配的 detections: 新建 track
        for di in unmatched_det:
            x1, y1, x2, y2, sc = detections[di]
            t = Track(
                track_id=self._next_id,
                bbox=(x1, y1, x2, y2),
                score=sc,
                age=0,
                hits=1,
                last_seen=self._frame_id,
                sample_count=0,
                next_sample_frame=self._frame_id,  # 立即可采
            )
            self._tracks.append(t)
            self._next_id += 1

        # 删除过老 tracks
        alive = [t for t in self._tracks if t.age <= self.max_age]
        if len(alive) != len(self._tracks):
            self._tracks = alive

        # 找出"该采脸"的 tracks (hits >= min_hits, 且到时间, 且未达上限)
        candidates: List[Track] = []
        for t in self._tracks:
            if t.hits < self.min_hits:
                continue
            if t.sample_count >= self.max_samples_per_track:
                continue
            if self._frame_id < t.next_sample_frame:
                continue
            candidates.append(t)
            t.sample_count += 1
            t.next_sample_frame = self._frame_id + self.sample_interval_frames

        return candidates

    @property
    def all_tracks(self) -> List[Track]:
        return list(self._tracks)

    @property
    def frame_id(self) -> int:
        return self._frame_id