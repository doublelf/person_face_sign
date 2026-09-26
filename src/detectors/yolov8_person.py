"""YOLOv8 行人检测 (仅 person 类) on Hailo-8。

Hailo Model Zoo yolov8n HEF 输出格式 (NMS 后处理已内嵌):
    output[batch_idx][class_id] -> ndarray (N, 5) 每行 [ymin, xmin, ymax, xmax, score],
    全部归一化到 [0, 1], 相对 640x640 letterbox 输入。

预处理:
  - letterbox (保持纵横比, 黑边 pad)
  - BGR -> RGB

后处理:
  - 只取 class 0 (person)
  - 反向 letterbox 回原图坐标 (x1, y1, x2, y2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .hailo_runner import HailoRunner, HailoMultiRunner
from typing import Optional  # noqa: F401  (also referenced below)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果。"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (原图坐标)
    score: float
    cls: int = 0  # COCO person


class YoloV8PersonDetector:
    """YOLOv8 行人检测器,包装 Hailo-8 上的 yolov8n HEF。

    支持两种用法:
      1) 单独使用 (M1): with YoloV8PersonDetector(hef) as det: ...
      2) 多模型共享 VDevice (M2+): 由 caller 自己持有 HailoMultiRunner,
         detector 接收一个 _SharedRunner 适配器。
    """

    PERSON_CLS = 0
    DEFAULT_INPUT_SIZE = (640, 640)
    DEFAULT_SCORE_THRESH = 0.40
    PAD_COLOR = (0, 0, 0)  # 参考官方 web_detection.py: 黑边

    def __init__(
        self,
        hef_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        score_thresh: float = DEFAULT_SCORE_THRESH,
        shared_runner=None,
        shared_model_name: str = "yolov8_person",
    ):
        self.hef_path = hef_path
        self.input_w, self.input_h = input_size
        self.score_thresh = score_thresh
        self._shared_runner = shared_runner
        self._shared_model_name = shared_model_name
        self._runner: HailoRunner | None = None
        self._input_layer = "yolov8n/input_layer1"
        self._output_layer = "yolov8n/yolov8_nms_postprocess"

    def __enter__(self) -> "YoloV8PersonDetector":
        if self._shared_runner is not None:
            slot = self._shared_runner.get_slot(self._shared_model_name)
            self._input_layer = slot.input_infos[0].name
            self._output_layer = slot.output_infos[0].name
            logger.info(
                "YoloV8PersonDetector (shared): in=%s out=%s score_thresh=%.2f",
                self._input_layer, self._output_layer, self.score_thresh,
            )
            return self
        if self.hef_path is None:
            raise ValueError("Either hef_path or shared_runner required")
        self._runner = HailoRunner(self.hef_path).__enter__()
        info = self._runner.input_infos[0]
        self._input_layer = info.name
        info = self._runner.output_infos[0]
        self._output_layer = info.name
        logger.info(
            "YoloV8PersonDetector ready: in=%s out=%s score_thresh=%.2f",
            self._input_layer, self._output_layer, self.score_thresh,
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
        """Letterbox resize + pad, 黑色填充, 返回 (canvas, scale, (pad_left, pad_top))。"""
        h, w = frame.shape[:2]
        scale = min(new_w / w, new_h / h)
        rw, rh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((new_h, new_w, 3), self_pad_color := 0, dtype=np.uint8)
        pad_l = (new_w - rw) // 2
        pad_t = (new_h - rh) // 2
        canvas[pad_t:pad_t + rh, pad_l:pad_l + rw] = resized
        return canvas, scale, (pad_l, pad_t)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """letterbox + BGR2RGB, 返回 (RGB canvas, scale, pad_l, pad_t)。"""
        canvas, scale, (pad_l, pad_t) = self._letterbox(frame_bgr, self.input_w, self.input_h)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return rgb, scale, pad_l, pad_t

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, _dt = self._do_run(x)
        return self._parse_person_output(out, h, w, scale, pad_l, pad_t)

    def detect_with_timing(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], float]:
        h, w = frame_bgr.shape[:2]
        rgb, scale, pad_l, pad_t = self._preprocess(frame_bgr)
        x = rgb[np.newaxis, ...]
        out, dt = self._do_run(x)
        return self._parse_person_output(out, h, w, scale, pad_l, pad_t), dt

    def _parse_person_output(
        self,
        out: dict,
        h: int,
        w: int,
        scale: float,
        pad_l: int,
        pad_t: int,
    ) -> List[Detection]:
        """解析 YOLOv8 NMS 输出,提取 person bbox。

        实际格式 (sensecraft / HailoRT 4.x yolov8n):
            out[name]               # dict
            out[name]               -> list[batch=1]
            out[name][0]            -> list[80 classes]
            out[name][0][cls]       -> ndarray (N, 5) 每行 [ymin, xmin, ymax, xmax, score]
        归一化到 [0, 1], 相对 640x640 letterbox 输入。
        """
        raw = out[self._output_layer]
        # batch dim
        if isinstance(raw, list):
            per_class = raw[0] if len(raw) > 0 and isinstance(raw[0], list) else raw
        elif isinstance(raw, np.ndarray):
            # 兼容密集 ndarray (batch, 80, max_dets, 5)
            per_class = raw[0] if raw.ndim == 3 else raw
        else:
            return []

        person = per_class[self.PERSON_CLS] if len(per_class) > self.PERSON_CLS else None
        if person is None:
            return []
        person = np.asarray(person, dtype=np.float32)
        if person.ndim != 2 or person.shape[1] != 5:
            return []
        scores = person[:, 4]
        mask = scores >= self.score_thresh
        if not np.any(mask):
            return []
        boxes = person[mask, :4]  # (N, 4): [ymin, xmin, ymax, xmax]
        scores_valid = scores[mask]

        # 转到 640x640 像素坐标
        y1_px = boxes[:, 0] * self.input_h - pad_t
        x1_px = boxes[:, 1] * self.input_w - pad_l
        y2_px = boxes[:, 2] * self.input_h - pad_t
        x2_px = boxes[:, 3] * self.input_w - pad_l
        # 反向 letterbox scale
        x1 = x1_px / scale
        y1 = y1_px / scale
        x2 = x2_px / scale
        y2 = y2_px / scale
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)
        return [
            Detection(
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                score=float(scores_valid[i]),
                cls=self.PERSON_CLS,
            )
            for i in range(len(scores_valid))
        ]
