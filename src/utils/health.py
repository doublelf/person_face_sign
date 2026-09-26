"""健康检查 + 异常处理工具。

封装:
- 摄像头断连自动重连
- 推理超时检测
- 异常计数器 (consecutive_failures)
- 看门狗心跳 (外部进程定期 ping)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class HealthMonitor:
    """运行时健康检查器。

    主要职责:
    - 跟踪连续失败次数, 超过阈值触发 alert
    - 提供 heartbeat (外部 watchdog 探测)
    - 记录最近异常
    """

    def __init__(
        self,
        max_consecutive_failures: int = 5,
        reconnect_interval_s: float = 2.0,
        inference_timeout_s: float = 30.0,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.reconnect_interval_s = reconnect_interval_s
        self.inference_timeout_s = inference_timeout_s
        self._consecutive_failures = 0
        self._total_failures = 0
        self._last_failure_ts: Optional[float] = None
        self._last_failure_reason: Optional[str] = None
        self._last_success_ts: Optional[float] = None
        self._heartbeat_ts: Optional[float] = None
        self._alert_callbacks: list[Callable[[str, dict], None]] = []
        self._lock = threading.Lock()

    def record_success(self, op: str = "default") -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._last_success_ts = time.time()

    def record_failure(self, reason: str, op: str = "default") -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._total_failures += 1
            self._last_failure_ts = time.time()
            self._last_failure_reason = reason
            logger.warning(
                "Health failure (consec=%d/%d total=%d): %s op=%s",
                self._consecutive_failures, self.max_consecutive_failures,
                self._total_failures, reason, op,
            )
            if self._consecutive_failures >= self.max_consecutive_failures:
                self._fire_alert("consecutive_failures", {
                    "consecutive": self._consecutive_failures,
                    "max": self.max_consecutive_failures,
                    "reason": reason,
                    "op": op,
                })

    def is_alive(self, heartbeat_max_age_s: float = 30.0) -> bool:
        """最近一次 heartbeat 是否在阈值内。"""
        if self._heartbeat_ts is None:
            return False
        return (time.time() - self._heartbeat_ts) < heartbeat_max_age_s

    def heartbeat(self) -> None:
        """Watchdog 调用, 表征主循环还在跑。"""
        with self._lock:
            self._heartbeat_ts = time.time()

    def add_alert_callback(self, cb: Callable[[str, dict], None]) -> None:
        with self._lock:
            self._alert_callbacks.append(cb)

    def _fire_alert(self, kind: str, info: dict) -> None:
        logger.error("ALERT: %s %s", kind, info)
        for cb in list(self._alert_callbacks):
            try:
                cb(kind, info)
            except Exception:
                logger.exception("Alert callback failed")

    def status(self) -> dict:
        with self._lock:
            return {
                "consecutive_failures": self._consecutive_failures,
                "total_failures": self._total_failures,
                "max_consecutive": self.max_consecutive_failures,
                "last_success_ts": self._last_success_ts,
                "last_failure_ts": self._last_failure_ts,
                "last_failure_reason": self._last_failure_reason,
                "heartbeat_ts": self._heartbeat_ts,
                "alive": self.is_alive(),
            }


# 全局单例
health = HealthMonitor()


class InferenceTimeout:
    """推理超时检测 context manager。"""

    def __init__(self, op_name: str, timeout_s: float, health_monitor=None):
        self.op_name = op_name
        self.timeout_s = timeout_s
        self.hm = health_monitor or health
        self._timer: Optional[threading.Timer] = None

    def __enter__(self):
        self._done = threading.Event()

        def on_timeout():
            if not self._done.is_set():
                self.hm.record_failure(f"inference timeout ({self.op_name})", self.op_name)

        self._timer = threading.Timer(self.timeout_s, on_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._done.set()
        if self._timer:
            self._timer.cancel()
        if exc_type is None:
            self.hm.record_success(self.op_name)
        else:
            self.hm.record_failure(f"{self.op_name}: {exc_val}", self.op_name)
        return False