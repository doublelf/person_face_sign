"""Pi Camera (CSI) FrameSource（picamera2 后端）。

使用 Raspberry Pi 官方 picamera2 库访问 CSI 摄像头,经 ISP 输出 BGR。
当用户接好 Pi Camera 后,通过本类启用。

Example:
    >>> src = PiCameraSource(out_queue=q, width=1280, height=720)
    >>> src.start()
"""
from __future__ import annotations

import logging
import queue

import numpy as np

from .base import FrameSource

logger = logging.getLogger(__name__)


class PiCameraSource(FrameSource):
    """Pi Camera (CSI) 帧源,使用 picamera2。"""

    def __init__(
        self,
        out_queue: "queue.Queue" = None,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 25,
    ):
        super().__init__(uri=f"picamera:{camera_id}", name="pi_cam")
        self.out_queue = out_queue
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cam = None
        self._frame_id = 0

    def _open(self) -> None:
        try:
            from picamera2 import Picamera2  # noqa: F401
        except ImportError as e:
            raise RuntimeError("picamera2 not installed") from e
        from picamera2 import Picamera2
        self._cam = Picamera2(camera_num=self.camera_id)
        cfg = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "BGR888"},
            controls={"FrameRate": self.fps},
        )
        self._cam.configure(cfg)
        self._cam.start()
        logger.info("Pi Camera %d started: %dx%d @ %d fps", self.camera_id, self.width, self.height, self.fps)
        self._frame_id = 0

    def _read_loop(self, on_frame) -> None:
        assert self._cam is not None
        while not self._stop_event.is_set():
            arr = self._cam.capture_array("main")
            if arr is None:
                continue
            frame = arr  # already BGR888
            fid = self._frame_id
            self._frame_id += 1
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put((fid, frame))

    def _close(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass
            self._cam = None