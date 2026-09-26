"""FrameSource: 抽象视频输入源。

设计原则:
- USB 摄像头 / Pi Camera / 本地视频 / 未来 RTSP 都走同一个接口
- 输出为统一的 (frame_id, BGR ndarray) 流
- 单线程拉流 + 解码，输出进无锁 SPSC 队列供下游消费
"""
from __future__ import annotations

import abc
import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class FrameSource(abc.ABC):
    """视频源抽象基类。"""

    def __init__(self, uri: str, name: str = "source"):
        self.uri = uri
        self.name = name
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._opened = False

    @abc.abstractmethod
    def _open(self) -> None:
        ...

    @abc.abstractmethod
    def _read_loop(self, on_frame) -> None:
        """子线程持续读帧并通过 on_frame(frame_id, bgr) 回调送出。"""
        ...

    @abc.abstractmethod
    def _close(self) -> None:
        ...

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("FrameSource already started")
        self._open()
        self._opened = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop_safe,
            name=f"FrameSource[{self.name}]",
            daemon=True,
        )
        self._thread.start()
        logger.info("FrameSource started: %s uri=%s", self.name, self.uri)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._close()
        self._opened = False
        logger.info("FrameSource stopped: %s", self.name)

    def _read_loop_safe(self) -> None:
        try:
            self._read_loop(lambda fid, frame: None)
        except Exception:
            logger.exception("FrameSource %s crashed", self.name)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self) -> None:
        self._stop_event.set()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event