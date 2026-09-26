"""视频帧采样器 (按时间间隔)。

策略: 用 ffmpeg `select='eq(n,...)'` 直接跳到目标时间点,
避免逐帧读+丢的低效做法。

API:
    sampler = VideoSampler(interval_s=1.0)
    for sample in sampler.iter(video_path):
        frame_id, timestamp, frame_bgr, source_name = sample
        ...

也支持多个视频合并采样:
    sampler = VideoSampler(interval_s=1.0)
    for sample in sampler.iter_multi([v1, v2, v3]):
        ...
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# (frame_id, timestamp_sec, bgr_ndarray, source_name)
Sample = Tuple[int, float, np.ndarray, str]


def _probe_duration(video_path: str) -> float:
    """用 ffprobe 拿视频时长 (秒)。失败返回 0。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception as e:
        logger.warning("ffprobe failed for %s: %s", video_path, e)
        return 0.0


def _seek_decode(video_path: str, ts: float, width: int, height: int) -> Optional[np.ndarray]:
    """ffmpeg -ss 跳到 ts, 输出 1 帧 BGR raw bytes。

    Returns: BGR ndarray (H, W, 3) 或 None
    """
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-ss", f"{ts:.3f}",
        "-i", video_path,
        "-an", "-dn", "-sn",
        "-vf", f"scale={width}:{height}",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "-frames:v", "1",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timeout at ts=%.2f", ts)
        return None
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in PATH")
    raw = proc.stdout
    if len(raw) != width * height * 3:
        # ffmpeg 没读到帧 (EOF / 关键帧问题), 返回 None
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()


class VideoSampler:
    """按时间间隔采帧。"""

    def __init__(
        self,
        interval_s: float = 1.0,
        width: int = 1280,
        height: int = 720,
        start_offset_s: float = 0.0,
        end_offset_s: float = 0.0,
    ):
        """
        Args:
            interval_s: 采样间隔 (秒)
            width/height: 输出帧尺寸
            start_offset_s: 从开头跳过 N 秒 (跳过片头)
            end_offset_s: 从结尾跳过 N 秒 (跳过片尾)
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.interval_s = interval_s
        self.width = width
        self.height = height
        self.start_offset_s = start_offset_s
        self.end_offset_s = end_offset_s

    def probe_duration(self, video_path: str) -> float:
        return _probe_duration(video_path)

    def iter(
        self,
        video_path: str,
        source_name: Optional[str] = None,
    ) -> Iterator[Sample]:
        """单个视频迭代采样。"""
        if source_name is None:
            source_name = Path(video_path).stem
        duration = self.probe_duration(video_path)
        if duration <= 0:
            logger.warning("Cannot determine duration of %s, skipping", video_path)
            return
        start = self.start_offset_s
        end = max(start, duration - self.end_offset_s)
        ts = start
        frame_id = 0
        # 安全 cap: 避免超长视频采太多
        max_frames = int((end - start) / self.interval_s) + 2
        while ts < end:
                frame = _seek_decode(video_path, ts, self.width, self.height)
                if frame is not None:
                    yield (frame_id, ts, frame, source_name)
                    frame_id += 1
                ts += self.interval_s
                max_frames -= 1
                if max_frames <= 0:
                    logger.warning("Max frames cap reached, stopping")
                    break

    def iter_multi(
        self,
        video_paths: List[str],
    ) -> Iterator[Sample]:
        """多视频依次采样, frame_id 全局递增。"""
        for vpath in video_paths:
            logger.info("Sampling video: %s", vpath)
            try:
                yield from self.iter(vpath)
            except Exception as e:
                logger.exception("Failed to sample %s: %s", vpath, e)