"""运行时监控与指标 (Prometheus 格式 + 内部计数器)。

设计:
- 全局 Metrics 单例, 各模块通过它更新
- 启动可选 HTTP exporter (端口默认 9090)
- 同时维护 internal dict 供内部断言 / 限速使用
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class Metrics:
    """轻量级指标收集 + Prometheus 输出。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._last_log_ts: dict[str, float] = {}
        self._started_at = time.time()
        self._prom_client = None

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms.setdefault(name, []).append(value)
            # 限制长度, 避免内存爆炸
            if len(self._histograms[name]) > 200:
                self._histograms[name] = self._histograms[name][-100:]

    def hist_avg(self, name: str) -> float:
        with self._lock:
            vals = self._histograms.get(name, [])
            return sum(vals) / len(vals) if vals else 0.0

    def get(self, name: str) -> float:
        with self._lock:
            return self._counters.get(name, self._gauges.get(name, 0.0))

    def summary(self) -> dict:
        with self._lock:
            out = {
                "uptime_s": time.time() - self._started_at,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
            }
            for k, vals in self._histograms.items():
                if vals:
                    out.setdefault("histograms", {})[k] = {
                        "avg": sum(vals) / len(vals),
                        "max": max(vals),
                        "n": len(vals),
                    }
            return out

    def render_prometheus(self) -> str:
        """输出 Prometheus 文本格式。"""
        lines = []
        with self._lock:
            for k, v in self._counters.items():
                lines.append(f"# TYPE {k} counter")
                lines.append(f"{k} {v}")
            for k, v in self._gauges.items():
                lines.append(f"# TYPE {k} gauge")
                lines.append(f"{k} {v}")
            for k, vals in self._histograms.items():
                if vals:
                    lines.append(f"# TYPE {k}_avg gauge")
                    lines.append(f"{k}_avg {sum(vals)/len(vals):.4f}")
                    lines.append(f"# TYPE {k}_max gauge")
                    lines.append(f"{k}_max {max(vals):.4f}")
        lines.append(f"# uptime")
        lines.append(f"uptime_s {time.time() - self._started_at:.4f}")
        return "\n".join(lines)

    def start_prometheus_exporter(self, port: int = 9090) -> Optional[threading.Thread]:
        """起一个 HTTP server 输出 /metrics。失败也不阻塞主程序。"""
        try:
            from prometheus_client import start_http_server
            start_http_server(port, registry=self._make_registry())
            logger.info("Prometheus exporter on :%d/metrics", port)
            return None
        except ImportError:
            logger.info("prometheus_client not available, skipping exporter")
            return None
        except Exception as e:
            logger.warning("Prometheus exporter failed: %s", e)
            return None

    def _make_registry(self):
        """构造一个临时 Registry, 把我们已有的 counters/gauges/histograms 注册进去."""
        try:
            from prometheus_client import (
                CollectorRegistry, Counter, Gauge, Summary,
            )
            reg = CollectorRegistry()
            with self._lock:
                for k, v in self._counters.items():
                    c = Counter(k, k.replace("_", " "), registry=reg)
                    c.inc(v)
                for k, v in self._gauges.items():
                    g = Gauge(k, k.replace("_", " "), registry=reg)
                    g.set(v)
                for k, vals in self._histograms.items():
                    if vals:
                        g_avg = Gauge(f"{k}_avg", f"{k} avg", registry=reg)
                        g_avg.set(sum(vals) / len(vals))
                        g_max = Gauge(f"{k}_max", f"{k} max", registry=reg)
                        g_max.set(max(vals))
            return reg
        except Exception:
            return None


# 全局单例
metrics = Metrics()


def update_resource_metrics() -> None:
    """更新 CPU/内存/NPU 等系统指标。"""
    try:
        import psutil
        proc = psutil.Process()
        mem = proc.memory_info().rss / 1024 / 1024
        metrics.set_gauge("process_memory_mb", mem)
        metrics.set_gauge("system_cpu_percent", psutil.cpu_percent(interval=None))
        metrics.set_gauge("system_memory_percent", psutil.virtual_memory().percent)
    except Exception:
        pass