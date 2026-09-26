"""人脸图像去重 (pHash 感知哈希)。

原理:
- 8x8 average hash (aHash) → 64-bit 二进制特征
- 不同光照/小扰动 → 相似 hash
- 完全相同图 → 完全相同 hash
- Hamming 距离 ≤ 4 bit ≈ 视觉上基本一致

API:
    hash1 = phash_8x8(face_112)
    distance = phash_distance(hash1, hash2)
    same_img = distance <= 4
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def phash_8x8(image_bgr: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """计算 8x8 平均哈希,返回 64-bit 整数。"""
    # 灰度
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr
    # 缩到 hash_size * highfreq_factor (32x32), DCT 前需要的尺寸
    resized = cv2.resize(gray, (hash_size * highfreq_factor, hash_size * highfreq_factor),
                          interpolation=cv2.INTER_AREA)
    # DCT
    dct = cv2.dct(np.float32(resized))
    # 取左上 8x8 低频
    dct_low = dct[:hash_size, :hash_size]
    # 计算均值 (除 DC 项)
    med = np.median(dct_low.flatten()[1:])
    # 大于均值为 1, 否则为 0
    bits = (dct_low > med).flatten()
    # 转为 64-bit int
    h = 0
    for i, b in enumerate(bits):
        if b:
            h |= (1 << i)
    return h


def phash_distance(h1: int, h2: int) -> int:
    """两 hash 间的 Hamming 距离 (0-64)。"""
    x = h1 ^ h2
    # Python 3.8+: int.bit_count()
    return bin(x).count("1") if hasattr(x, "bit_count") is False else x.bit_count()


def phash_similar(h1: int, h2: int, max_distance: int = 4) -> bool:
    """Hamming 距离 ≤ max_distance 视为同图。"""
    return phash_distance(h1, h2) <= max_distance