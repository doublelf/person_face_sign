"""5 点人脸对齐到 112×112。

参考 ArcFace / insightface 标准模板:

  模板 landmark (112×112 坐标系):
    左眼 (0):  (38.2946, 51.6963)
    右眼 (1):  (73.5318, 51.5014)
    鼻  (2):   (56.0252, 71.7366)
    左嘴角 (3):(41.5493, 92.3655)
    右嘴角 (4):(70.7299, 92.2041)

  由源 5 点 + 模板 5 点, 用相似变换 (scale + rotation + translation)
  估计 transform,然后 warp 到 112×112 输出。
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ArcFace 5-pt template (112x112)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face_5pt(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    output_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """用 5 点 landmark 把人脸对齐到 output_size (默认 112×112)。

    Args:
        image_bgr: BGR 图像
        landmarks: (5, 2) ndarray, 原图坐标系的 5 个点 (顺序: 左眼, 右眼, 鼻, 左嘴角, 右嘴角)
        output_size: 输出尺寸

    Returns:
        (H, W, 3) uint8 BGR 对齐后的人脸图
    """
    assert landmarks.shape == (5, 2), f"landmarks must be (5, 2), got {landmarks.shape}"
    template = ARCFACE_TEMPLATE.copy()
    if output_size != (112, 112):
        sx = output_size[0] / 112.0
        sy = output_size[1] / 112.0
        template = template * np.array([sx, sy], dtype=np.float32)

    # Estimate similarity transform: dst = template, src = landmarks
    # cv2.estimateAffinePartial2D expects src, dst
    M, _ = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32),
        template,
        method=cv2.LMEDS,
    )
    if M is None:
        # Fallback: 用纯平移 (不够准但保证有输出)
        cx_src = landmarks[:, 0].mean()
        cy_src = landmarks[:, 1].mean()
        cx_dst = template[:, 0].mean()
        cy_dst = template[:, 1].mean()
        M = np.array([[1.0, 0.0, cx_dst - cx_src],
                      [0.0, 1.0, cy_dst - cy_src]], dtype=np.float32)
    aligned = cv2.warpAffine(
        image_bgr, M, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned