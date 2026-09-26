"""USB UVC 摄像头 FrameSource（V4L2 直通）。

使用 OpenCV VideoCapture + V4L2 后端。摄像头输出 MJPEG/YUYV 直通,
不需 ffmpeg 解码。

Example:
    >>> src = USBCameraSource("/dev/video0", out_queue=q, width=1280, height=720)
    >>> src.start()
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import cv2
import numpy as np

from .base import FrameSource

logger = logging.getLogger(__name__)


class USBCameraSource(FrameSource):
    """USB 摄像头帧源,使用 OpenCV + V4L2 后端。"""

    def __init__(
        self,
        device: str = "/dev/video0",
        out_queue: "queue.Queue" = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
        warmup_frames: int = 10,
    ):
        super().__init__(uri=device, name="usb_cam")
        self.device = device
        self.out_queue = out_queue
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames

        self._cap: cv2.VideoCapture | None = None
        self._frame_id = 0

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warmup to drop initial frames
        for _ in range(self.warmup_frames):
            self._cap.read()
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "USB camera %s opened: %dx%d @ %.1f fps",
            self.device, actual_w, actual_h, actual_fps,
        )
        self.width, self.height = actual_w, actual_h
        self._frame_id = 0

    def _read_loop(self, on_frame) -> None:
        assert self._cap is not None
        cap = self._cap
        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("Camera read failed, retrying")
                time.sleep(0.05)
                continue
            fid = self._frame_id
            self._frame_id += 1
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put((fid, frame))

    def _close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None