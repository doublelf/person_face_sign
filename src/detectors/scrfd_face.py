"""SCRFD 人脸检测器 (Hailo-8 上的 SCRFD-10G HEF)。

输出格式 (来自 sensecraft/HailoRT 4.x yolov8n 一致的 NMS-postprocess 风格,
但 SCRFD-10G 保留原始 3 个 FPN 头未做 NMS,所以由我们自己解码):

  score_head:   (1, H, W, 2)   # 2 anchors × 1 face-score
  bbox_head:    (1, H, W, 8)   # 2 anchors × 4 bbox deltas (dx, dy, dw, dh)
  kps_head:     (1, H, W, 20)  # 2 anchors × 10 keypoint deltas (5 pts × 2 coords)

每个 FPN 层对应一个 stride (8/16/32), 三层 head 名固定:
  stride=8:  conv41 / conv42 / conv43
  stride=16: conv49 / conv50 / conv51
  stride=32: conv56 / conv57 / conv58

解码公式:
  anchor_cx = (j + 0.5) * stride
  anchor_cy = (i + 0.5) * stride
  bbox_cx = anchor_cx + dx * stride
  bbox_cy = anchor_cy + dy * stride
  bbox_w  = stride * exp(dw)
  bbox_h  = stride * exp(dh)
  kp_x    = anchor_cx + kx * stride
  kp_y    = anchor_cy + ky * stride

后处理:
  1. 三层所有 anchor 解码 + sigmoid score
  2. 按 score 阈值过滤
  3. NMS (类内 NMS,同一个人脸只有一个 bbox)
  4. 返回 top-K
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .hailo_runner import HailoRunner

logger = logging.getLogger(__name__)


# (stride, score_head_name, bbox_head_name, kps_head_name)
SCRFD_HEADS = [
    (8,  "scrfd_10g/conv41", "scrfd_10g/conv42", "scrfd_10g/conv43"),
    (16, "scrfd_10g/conv49", "scrfd_10g/conv50", "scrfd_10g/conv51"),
    (32, "scrfd_10g/conv56", "scrfd_10g/conv57", "scrfd_10g/conv58"),
]


@dataclass
class Face:
    """单个人脸检测结果。"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (输入图坐标)
    score: float
    landmarks: np.ndarray  # (5, 2) 五点坐标 (输入图坐标)


class SCRFDFaceDetector:
    """SCRFD-10G 人脸检测器,支持全图或裁剪区域输入。"""

    DEFAULT_INPUT_SIZE = (640, 640)
    DEFAULT_SCORE_THRESH = 0.50
    DEFAULT_NMS_IOU_THRESH = 0.50
    DEFAULT_TOP_K = 50
    DEFAULT_PRE_NMS_TOPK = 1000  # 限制 NMS 输入 boxes 数量

    def __init__(
        self,
        hef_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        score_thresh: float = DEFAULT_SCORE_THRESH,
        nms_iou_thresh: float = DEFAULT_NMS_IOU_THRESH,
        top_k: int = DEFAULT_TOP_K,
        pre_nms_topk: int = DEFAULT_PRE_NMS_TOPK,
        shared_runner=None,
        shared_model_name: str = "scrfd_face",
    ):
        self.hef_path = hef_path
        self.input_w, self.input_h = input_size
        self.score_thresh = score_thresh
        self.nms_iou_thresh = nms_iou_thresh
        self.top_k = top_k
        self.pre_nms_topk = pre_nms_topk
        self._shared_runner = shared_runner
        self._shared_model_name = shared_model_name
        self._runner: HailoRunner | None = None
        self._input_layer = "scrfd_10g/input_layer1"

    def __enter__(self) -> "SCRFDFaceDetector":
        if self._shared_runner is not None:
            slot = self._shared_runner.get_slot(self._shared_model_name)
            self._input_layer = slot.input_infos[0].name
            logger.info(
                "SCRFDFaceDetector (shared): in=%s score_thresh=%.2f nms_iou=%.2f",
                self._input_layer, self.score_thresh, self.nms_iou_thresh,
            )
            return self
        if self.hef_path is None:
            raise ValueError("Either hef_path or shared_runner required")
        self._runner = HailoRunner(self.hef_path).__enter__()
        self._input_layer = self._runner.input_infos[0].name
        logger.info(
            "SCRFDFaceDetector ready: in=%s score_thresh=%.2f nms_iou=%.2f",
            self._input_layer, self.score_thresh, self.nms_iou_thresh,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._runner is not None:
            self._runner.__exit__(exc_type, exc, tb)
            self._runner = None

    def _do_run(self, x):
        if self._shared_runner is not None:
            return self._shared_runner.run_with_timing(self._shared_model_name, {self._input_layer: x})
        return self._runner.run_with_timing({self._input_layer: x})

    @staticmethod
    def _letterbox(frame: np.ndarray, new_w: int, new_h: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        scale = min(new_w / w, new_h / h)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_h, new_w, 3), 0, dtype=np.uint8)
        pad_l = (new_w - rw) // 2
        pad_t = (new_h - rh) // 2
        canvas[pad_t:pad_t + rh, pad_l:pad_l + rw] = resized
        return canvas, scale, (pad_l, pad_t)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int, int, int]:
        canvas, scale, (pad_l, pad_t) = self._letterbox(frame_bgr, self.input_w, self.input_h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        return rgb, scale, pad_l, pad_t, h, w

    def _decode(self, outputs: dict) -> np.ndarray:
        """把三层 head 解码为 (M, 15) ndarray:
        每行 [x1, y1, x2, y2, score, kp1_x, kp1_y, ..., kp5_x, kp5_y]
        坐标相对 letterbox 后的 640x640 输入。
        """
        proposals: List[np.ndarray] = []
        for stride, s_name, b_name, k_name in SCRFD_HEADS:
            s_head = np.asarray(outputs[s_name])[0]  # (H, W, 2)
            b_head = np.asarray(outputs[b_name])[0]  # (H, W, 8)
            k_head = np.asarray(outputs[k_name])[0]  # (H, W, 20)
            H, W, _ = s_head.shape

            s_sig = 1.0 / (1.0 + np.exp(-s_head))
            mask = s_sig >= self.score_thresh
            if not mask.any():
                continue

            flat_mask = mask.reshape(-1)
            flat_score = s_sig.reshape(-1)
            flat_bbox = b_head.reshape(-1, 8)
            flat_kps = k_head.reshape(-1, 20)

            sc = flat_score[flat_mask]
            idx = np.flatnonzero(flat_mask)
            a = (idx % 2).astype(np.int32)
            cell = (idx // 2).astype(np.int32)
            ii = cell // W
            jj = cell % W

            d = np.empty((len(idx), 4), dtype=np.float32)
            sel_a0 = a == 0
            sel_a1 = ~sel_a0
            if sel_a0.any():
                d[sel_a0] = flat_bbox[cell[sel_a0], 0:4]
            if sel_a1.any():
                d[sel_a1] = flat_bbox[cell[sel_a1], 4:8]

            kp_arr = np.empty((len(idx), 10), dtype=np.float32)
            if sel_a0.any():
                kp_arr[sel_a0] = flat_kps[cell[sel_a0], 0:10]
            if sel_a1.any():
                kp_arr[sel_a1] = flat_kps[cell[sel_a1], 10:20]

            cx_a = (jj.astype(np.float32) + 0.5) * stride
            cy_a = (ii.astype(np.float32) + 0.5) * stride

            x_c = cx_a + d[:, 0] * stride
            y_c = cy_a + d[:, 1] * stride
            w_b = stride * np.exp(d[:, 2])
            h_b = stride * np.exp(d[:, 3])
            x1 = x_c - w_b / 2
            y1 = y_c - h_b / 2
            x2 = x_c + w_b / 2
            y2 = y_c + h_b / 2

            kps_decoded = np.empty_like(kp_arr)
            for k in range(5):
                kps_decoded[:, 2 * k]     = cx_a + kp_arr[:, 2 * k]     * stride
                kps_decoded[:, 2 * k + 1] = cy_a + kp_arr[:, 2 * k + 1] * stride

            props = np.concatenate([
                np.stack([x1, y1, x2, y2, sc], axis=1),
                kps_decoded,
            ], axis=1)
            proposals.append(props)

        if not proposals:
            return np.zeros((0, 15), dtype=np.float32)
        all_props = np.concatenate(proposals, axis=0)
        # 关键优化: NMS 之前先按 score 取 top-K, 避免 16k boxes 的 O(N^2) NMS
        if len(all_props) > self.pre_nms_topk:
            order = np.argpartition(-all_props[:, 4], self.pre_nms_topk - 1)[:self.pre_nms_topk]
            all_props = all_props[order]
        return all_props

    @staticmethod
    def _nms(boxes: np.ndarray, iou_thresh: float) -> List[int]:
        """类内 NMS, 返回保留下来的 index (按 score 降序)。"""
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        sc = boxes[:, 4]
        areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))

        order = np.argsort(-sc, kind="stable")
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            keep_mask = iou <= iou_thresh
            order = rest[keep_mask]
        return keep

    @staticmethod
    def _nms(boxes: np.ndarray, iou_thresh: float) -> List[int]:
        """类内 NMS, 返回保留下来的 index (按 score 降序)。

        完全向量化实现, 处理 16k+  boxes 也要在 < 50 ms 内完成。
        """
        if len(boxes) == 0:
            return []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        sc = boxes[:, 4]
        areas = np.maximum(0.0, (x2 - x1)) * np.maximum(0.0, (y2 - y1))

        # 按 score 降序
        order = np.argsort(-sc, kind="stable")
        keep: List[int] = []
        suppressed = np.zeros(len(boxes), dtype=bool)
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            # 计算与 i 的 IoU
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            keep_mask = iou <= iou_thresh
            order = rest[keep_mask]
        return keep

    def detect(self, frame_bgr: np.ndarray) -> List[Face]:
        """对单帧做全图人脸检测。"""
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t, h_orig, w_orig = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, _dt = self._do_run(x)
        return self._faces_from_outputs(out, scale, pad_l, pad_t, h_orig, w_orig)

    def detect_with_timing(self, frame_bgr: np.ndarray) -> Tuple[List[Face], float]:
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t, h_orig, w_orig = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, dt = self._do_run(x)
        return self._faces_from_outputs(out, scale, pad_l, pad_t, h_orig, w_orig), dt

    def _faces_from_outputs(self, out, scale, pad_l, pad_t, h_orig, w_orig):
        props = self._decode(out)  # (M, 15) in 640x640
        if len(props) == 0:
            return []
        keep = self._nms(props, self.nms_iou_thresh)
        props = props[keep]
        if len(props) > self.top_k:
            order = props[:, 4].argsort()[::-1][:self.top_k]
            props = props[order]
        faces: List[Face] = []
        for p in props:
            x1 = max(0.0, (p[0] - pad_l) / scale)
            y1 = max(0.0, (p[1] - pad_t) / scale)
            x2 = min(float(w_orig - 1), (p[2] - pad_l) / scale)
            y2 = min(float(h_orig - 1), (p[3] - pad_t) / scale)
            score = float(p[4])
            landmarks = np.zeros((5, 2), dtype=np.float32)
            for k in range(5):
                landmarks[k, 0] = (p[5 + 2 * k] - pad_l) / scale
                landmarks[k, 1] = (p[5 + 2 * k + 1] - pad_t) / scale
            faces.append(Face(bbox=(x1, y1, x2, y2), score=score, landmarks=landmarks))
        return faces

    def detect_in_crop(self, frame_bgr: np.ndarray, crop_bbox: Tuple[float, float, float, float]) -> List[Face]:
        """在给定裁剪区域 (原图坐标) 内检人脸, 返回的人脸坐标是相对原图的 (不是相对 crop)。"""
        x1, y1, x2, y2 = crop_bbox
        h_full, w_full = frame_bgr.shape[:2]
        pad_w = int((x2 - x1) * 0.1)
        pad_h = int((y2 - y1) * 0.1)
        cx1 = max(0, int(x1 - pad_w))
        cy1 = max(0, int(y1 - pad_h))
        cx2 = min(w_full, int(x2 + pad_w))
        cy2 = min(h_full, int(y2 + pad_h))
        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return []
        crop = frame_bgr[cy1:cy2, cx1:cx2]
        faces = self.detect(crop)
        for f in faces:
            fx1, fy1, fx2, fy2 = f.bbox
            f.bbox = (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1)
            f.landmarks[:, 0] += cx1
            f.landmarks[:, 1] += cy1
        return faces