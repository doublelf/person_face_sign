"""本地视频文件 FrameSource。

使用 ffmpeg 子进程解码,输出 BGR24 原始帧到 stdout。
子线程负责 read + decode,主线程 / 上层消费者从队列取帧。

注意:
- 默认不做实时限速 (不用 -re),让 ffmpeg 尽量快地解码,瓶颈通常在推理而非解码
- 使用阻塞读 + select() 等待数据,避免非阻塞读误判 EOF
- 当视频编码是 HEVC 时,优先用 hevc_v4l2m2m 走 RPi 5 硬件解码

Example:
    >>> src = VideoFileSource("/path/to/video.mp4", queue=frame_queue)
    >>> src.start()
"""
from __future__ import annotations

import logging
import os
import queue
import select
import subprocess
import threading
from typing import Optional

import numpy as np

from .base import FrameSource

logger = logging.getLogger(__name__)

_BYTES_PER_PIXEL = 3
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_QUEUE_SIZE = 4


class VideoFileSource(FrameSource):
    """读取本地视频文件,通过 ffmpeg 解码后输出 BGR 帧。"""

    def __init__(
        self,
        path: str,
        out_queue: "queue.Queue",
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps_hint: float = 0.0,
        use_realtime: bool = False,
        loop: bool = False,
        read_timeout_s: float = 2.0,
    ):
        super().__init__(uri=path, name="video_file")
        self.path = path
        self.out_queue = out_queue
        self.width = width
        self.height = height
        self.fps_hint = fps_hint
        self.use_realtime = use_realtime
        self.loop = loop
        self.read_timeout_s = read_timeout_s

        self._proc: Optional[subprocess.Popen] = None
        self._frame_id = 0
        self._frame_bytes = width * height * _BYTES_PER_PIXEL
        self._loop_count = 0
        self._codec: str = ""

    def _probe_codec(self) -> str:
        """ffprobe 出输入视频的编码格式。"""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name",
                 "-of", "default=nw=1:nk=1", self.path],
                capture_output=True, text=True, timeout=5,
            )
            codec = r.stdout.strip()
            return codec or "unknown"
        except Exception as e:
            logger.warning("ffprobe failed: %s", e)
            return "unknown"

    def _build_cmd(self) -> list[str]:
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error"]
        # 硬件解码: 仅 HEVC 可用 RPi 的 hevc_v4l2m2m
        if self._codec == "hevc":
            cmd += ["-c:v", "hevc_v4l2m2m"]
        # 软件解码 (H.264 等) 走默认 codec,不再加 hwaccel 标志
        if self.use_realtime:
            cmd += ["-re"]
        cmd += ["-i", self.path]
        cmd += [
            "-an", "-dn", "-sn",
            "-vf", f"scale={self.width}:{self.height}",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
        ]
        if self.fps_hint > 0:
            cmd += ["-r", str(self.fps_hint)]
        cmd += ["pipe:1"]
        return cmd

    def _open(self) -> None:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Video file not found: {self.path}")
        self._codec = self._probe_codec()
        cmd = self._build_cmd()
        logger.info("Opening video file (codec=%s): ffmpeg %s",
                    self._codec, " ".join(cmd[3:]))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH")
        self._frame_id = 0
        self._loop_count = 0

    def _wait_for_data(self, fileno: int, timeout: float) -> bool:
        """Wait until fileno has data or timeout. Returns False on EOF."""
        r, _, _ = select.select([fileno], [], [], timeout)
        if not r:
            return True  # timed out but not EOF
        return True

    def _read_loop(self, on_frame) -> None:
        assert self._proc is not None
        stdout = self._proc.stdout
        fd = stdout.fileno()
        frame_size = self._frame_bytes
        leftover = b""

        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="VideoFileSource.stderr",
        )
        stderr_thread.start()

        while not self._stop_event.is_set():
            # Wait for data availability
            self._wait_for_data(fd, self.read_timeout_s)
            try:
                chunk = os.read(fd, frame_size - len(leftover))
            except BlockingIOError:
                continue
            if not chunk:
                # EOF or pipe closed
                if self.loop and not self._stop_event.is_set():
                    self._loop_count += 1
                    logger.info("Video EOF, looping (count=%d)", self._loop_count)
                    try:
                        self._proc.wait(timeout=1.0)
                    except Exception:
                        pass
                    self._proc = None
                    try:
                        self._proc = subprocess.Popen(
                            self._build_cmd(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=0,
                        )
                        stdout = self._proc.stdout
                        fd = stdout.fileno()
                    except Exception:
                        logger.exception("Failed to reopen ffmpeg on loop")
                        break
                    leftover = b""
                    continue
                else:
                    logger.info("Video EOF reached")
                    break
            leftover += chunk
            while len(leftover) >= frame_size and not self._stop_event.is_set():
                raw = leftover[:frame_size]
                leftover = leftover[frame_size:]
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, _BYTES_PER_PIXEL)
                    ).copy()
                except ValueError:
                    logger.warning("Frame reshape failed, skipping")
                    continue
                fid = self._frame_id
                self._frame_id += 1
                if self.out_queue.full():
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.out_queue.put((fid, frame))

    def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        for line in iter(self._proc.stderr.readline, b""):
            if line:
                logger.debug("[ffmpeg] %s", line.decode("utf-8", "replace").rstrip())

    def _close(self) -> None:
        if self._proc is not None:
            try:
                self._proc.stdout.close()
            except Exception:
                pass
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=1.0)
            except Exception:
                pass
            self._proc = None