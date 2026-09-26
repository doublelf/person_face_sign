"""ArcFace-MobileFaceNet 512-d embedding 提取。

输入: 112×112×3 UINT8 对齐人脸
输出: 512-dim float32 embedding (需要 L2 normalize 后用于 cosine 相似度)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .hailo_runner import HailoRunner

logger = logging.getLogger(__name__)


class ArcFaceEmbedder:
    """ArcFace-MobileFaceNet 512-dim embedding 包装。

    输入必须是已经对齐到 112×112 的人脸图 (与训练时 ArcFace 模板一致)。
    输出做 L2 normalize 后用 cosine 相似度比较。
    """

    DEFAULT_INPUT_SIZE = (112, 112)
    OUTPUT_DIM = 512

    def __init__(
        self,
        hef_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        shared_runner=None,
        shared_model_name: str = "arcface_embed",
    ):
        self.hef_path = hef_path
        self.input_w, self.input_h = input_size
        self._shared_runner = shared_runner
        self._shared_model_name = shared_model_name
        self._runner: HailoRunner | None = None
        self._input_layer = "arcface_mobilefacenet/input_layer1"

    def __enter__(self) -> "ArcFaceEmbedder":
        if self._shared_runner is not None:
            slot = self._shared_runner.get_slot(self._shared_model_name)
            self._input_layer = slot.input_infos[0].name
            logger.info("ArcFaceEmbedder (shared) ready: in=%s", self._input_layer)
            return self
        if self.hef_path is None:
            raise ValueError("Either hef_path or shared_runner required")
        self._runner = HailoRunner(self.hef_path).__enter__()
        self._input_layer = self._runner.input_infos[0].name
        logger.info("ArcFaceEmbedder ready: in=%s", self._input_layer)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._runner is not None:
            self._runner.__exit__(exc_type, exc, tb)
            self._runner = None

    def _do_run(self, x):
        if self._shared_runner is not None:
            return self._shared_runner.run_with_timing(self._shared_model_name, {self._input_layer: x})
        return self._runner.run_with_timing({self._input_layer: x})

    def embed(self, face_bgr_112: np.ndarray, normalize: bool = True) -> np.ndarray:
        """对单张 112×112 人脸取 512-d embedding。

        Args:
            face_bgr_112: HxWx3 uint8, 已对齐到 112×112 (本项目用 face_align.align_face_5pt)
            normalize: 是否 L2 normalize (cosine 相似度场景需要 True)

        Returns:
            (512,) float32 向量
        """
        if face_bgr_112.shape[:2] != (self.input_h, self.input_w):
            face_bgr_112 = cv2.resize(face_bgr_112, (self.input_w, self.input_h))
        # ArcFace 训练用 RGB; 我们的 align_face_5pt 输出 BGR, 转换
        rgb = cv2.cvtColor(face_bgr_112, cv2.COLOR_BGR2RGB)
        x = rgb[np.newaxis, ...]
        result, _dt = self._do_run(x)
        arr = list(result.values())[0][0]  # (512,)
        if normalize:
            n = np.linalg.norm(arr)
            if n > 1e-9:
                arr = arr / n
        return arr.astype(np.float32)

    def embed_batch(self, faces_bgr_112: List[np.ndarray], normalize: bool = True) -> np.ndarray:
        """批量 embedding。

        Returns:
            (N, 512) float32
        """
        if not faces_bgr_112:
            return np.zeros((0, self.OUTPUT_DIM), dtype=np.float32)
        batch_rgb = np.stack([
            cv2.cvtColor(f if f.shape[:2] == (self.input_h, self.input_w)
                         else cv2.resize(f, (self.input_w, self.input_h)),
                         cv2.COLOR_BGR2RGB)
            for f in faces_bgr_112
        ], axis=0)
        result, _dt = self._do_run(batch_rgb)
        arr = list(result.values())[0]  # (N, 512)
        if normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            arr = arr / norms
        return arr.astype(np.float32)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        """两个 L2-normalized 向量的 cosine 距离 (1 - cos_sim)。"""
        return float(1.0 - np.dot(a, b))