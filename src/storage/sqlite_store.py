"""SQLite 持久化 (clusters + appearances)。

设计要点:
- WAL 模式: 允许边写边读,性能更好
- 简单 DAO,不需要 ORM
- 支持 schema 自动初始化
- 异步访问: 暂不开线程,简单 critical section 保护

Schema (严格按 PRD):
    clusters(cluster_id PK, name, label, first_seen, last_seen,
             count, best_emb_path, best_face_path, model_version)
    appearances(appearance_id PK, cluster_id FK, timestamp,
                track_id, pedestrian_bbox, face_bbox, embedding_path,
                quality_score, model_version)
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    name TEXT,
    label TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    best_emb_path TEXT,
    best_face_path TEXT,
    model_version TEXT
);

CREATE TABLE IF NOT EXISTS appearances (
    appearance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    track_id INTEGER,
    pedestrian_bbox TEXT,
    face_bbox TEXT,
    embedding_path TEXT,
    quality_score REAL,
    model_version TEXT,
    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
);

CREATE INDEX IF NOT EXISTS idx_app_cluster ON appearances(cluster_id);
CREATE INDEX IF NOT EXISTS idx_app_ts ON appearances(timestamp);
"""


class ClusterRow:
    def __init__(self, cluster_id: int, name: Optional[str], label: Optional[str],
                 first_seen: float, last_seen: float, count: int,
                 best_emb_path: Optional[str], best_face_path: Optional[str],
                 model_version: Optional[str]):
        self.cluster_id = cluster_id
        self.name = name
        self.label = label
        self.first_seen = first_seen
        self.last_seen = last_seen
        self.count = count
        self.best_emb_path = best_emb_path
        self.best_face_path = best_face_path
        self.model_version = model_version

    def to_dict(self):
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "label": self.label,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
            "best_emb_path": self.best_emb_path,
            "best_face_path": self.best_face_path,
            "model_version": self.model_version,
        }


class AppearanceRow:
    def __init__(self, appearance_id: int, cluster_id: int, timestamp: float,
                 track_id: Optional[int], pedestrian_bbox, face_bbox,
                 embedding_path: Optional[str], quality_score: Optional[float],
                 model_version: Optional[str]):
        self.appearance_id = appearance_id
        self.cluster_id = cluster_id
        self.timestamp = timestamp
        self.track_id = track_id
        self.pedestrian_bbox = pedestrian_bbox
        self.face_bbox = face_bbox
        self.embedding_path = embedding_path
        self.quality_score = quality_score
        self.model_version = model_version


class SQLiteStore:
    """SQLite 存储, 提供 cluster + appearance 的 CRUD."""

    def __init__(self, db_path: str, model_version: str = "v1"):
        self.db_path = db_path
        self.model_version = model_version
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        # 下一个 cluster_id
        self._next_cluster_id = self._compute_next_cluster_id()
        logger.info("SQLiteStore: db=%s next_cluster_id=%d", db_path, self._next_cluster_id)

    def _compute_next_cluster_id(self) -> int:
        cur = self._conn.execute("SELECT COALESCE(MAX(cluster_id), 0) FROM clusters")
        row = cur.fetchone()
        return int(row[0]) + 1

    @property
    def next_cluster_id(self) -> int:
        return self._next_cluster_id

    def upsert_cluster(
        self,
        cluster_id: int,
        first_seen: float,
        last_seen: float,
        count_delta: int = 1,
        best_emb_path: Optional[str] = None,
        best_face_path: Optional[str] = None,
    ) -> None:
        with self._lock:
            existing = self._conn.execute(
                "SELECT count, best_emb_path, best_face_path FROM clusters WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO clusters(cluster_id, first_seen, last_seen, count, "
                    "best_emb_path, best_face_path, model_version) VALUES (?,?,?,?,?,?,?)",
                    (cluster_id, first_seen, last_seen, count_delta,
                     best_emb_path, best_face_path, self.model_version),
                )
            else:
                # 累加 count, 更新 last_seen
                cur_count = existing[0]
                self._conn.execute(
                    "UPDATE clusters SET count=?, last_seen=?, "
                    "best_emb_path=COALESCE(?, best_emb_path), "
                    "best_face_path=COALESCE(?, best_face_path) "
                    "WHERE cluster_id=?",
                    (cur_count + count_delta, last_seen,
                     best_emb_path, best_face_path, cluster_id),
                )
            self._conn.commit()

    def add_appearance(
        self,
        cluster_id: int,
        timestamp: float,
        track_id: Optional[int],
        pedestrian_bbox: Tuple[float, float, float, float],
        face_bbox: Tuple[float, float, float, float],
        embedding_path: Optional[str],
        quality_score: Optional[float],
    ) -> int:
        # 兼容 numpy 标量
        def _f(x):
            return float(x)
        ped = [_f(v) for v in pedestrian_bbox]
        fce = [_f(v) for v in face_bbox]
        qs = _f(quality_score) if quality_score is not None else None
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO appearances(cluster_id, timestamp, track_id, "
                "pedestrian_bbox, face_bbox, embedding_path, quality_score, model_version) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (cluster_id, timestamp, track_id,
                 json.dumps(ped),
                 json.dumps(fce),
                 embedding_path,
                 qs,
                 self.model_version),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def allocate_cluster_id(self) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    def list_clusters(self, order_by: str = "count DESC", limit: int = 100) -> List[ClusterRow]:
        with self._lock:
            cur = self._conn.execute(
                f"SELECT cluster_id, name, label, first_seen, last_seen, count, "
                f"best_emb_path, best_face_path, model_version FROM clusters "
                f"ORDER BY {order_by} LIMIT ?",
                (limit,),
            )
            return [ClusterRow(*row) for row in cur.fetchall()]

    def get_cluster(self, cluster_id: int) -> Optional[ClusterRow]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT cluster_id, name, label, first_seen, last_seen, count, "
                "best_emb_path, best_face_path, model_version FROM clusters WHERE cluster_id=?",
                (cluster_id,),
            )
            row = cur.fetchone()
            return ClusterRow(*row) if row else None

    def set_cluster_name(self, cluster_id: int, name: str, label: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE clusters SET name=?, label=COALESCE(?, label) WHERE cluster_id=?",
                (name, label, cluster_id),
            )
            self._conn.commit()

    def merge_clusters(self, src_id: int, dst_id: int) -> int:
        """合并 src cluster 到 dst。返回被移动的 appearance 数量。

        规则:
        - dst 保留原 cluster_id, 累加 count
        - src 所有 appearances.cluster_id 改为 dst
        - src.cluster 保留 (count=0), 但会被 list_clusters 按 count DESC 自然排到末尾
        """
        with self._lock:
            # 1) 累加 dst 的 count (从 src 的 count)
            cur = self._conn.execute(
                "SELECT count FROM clusters WHERE cluster_id=?",
                (dst_id,),
            ).fetchone()
            if cur is None:
                return 0
            cur = self._conn.execute(
                "SELECT count FROM clusters WHERE cluster_id=?",
                (src_id,),
            ).fetchone()
            if cur is None:
                return 0
            src_count = int(cur[0])
            self._conn.execute(
                "UPDATE clusters SET count = count + ?, last_seen = MAX(last_seen, "
                "(SELECT last_seen FROM (SELECT last_seen FROM clusters WHERE cluster_id=?) t)) "
                "WHERE cluster_id=?",
                (src_count, src_id, dst_id),
            )
            # 2) 把 src 的所有 appearances 移到 dst
            cur = self._conn.execute(
                "UPDATE appearances SET cluster_id=? WHERE cluster_id=?",
                (dst_id, src_id),
            )
            moved = cur.rowcount
            # 3) 把 src 的 count 清零
            self._conn.execute(
                "UPDATE clusters SET count=0 WHERE cluster_id=?",
                (src_id,),
            )
            self._conn.commit()
            return moved

    def delete_cluster(self, cluster_id: int) -> int:
        """删除 cluster (含 appearances)。返回删除 appearance 数。"""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM appearances WHERE cluster_id=?",
                (cluster_id,),
            )
            n_app = cur.rowcount
            self._conn.execute(
                "DELETE FROM clusters WHERE cluster_id=?",
                (cluster_id,),
            )
            self._conn.commit()
            return n_app

    def list_embeddings(self, cluster_id: Optional[int] = None) -> list:
        """从 appearance 表里抽 embedding_path + cluster_id。

        注意: 当前实现不存 embedding_path 实际内容 (存到磁盘的话占空间)。
        本方法返回 (appearance_id, cluster_id, timestamp) 元组列表,
        用于离线重聚类时按需从外部 embedding 存储读。
        """
        with self._lock:
            if cluster_id is None:
                cur = self._conn.execute(
                    "SELECT appearance_id, cluster_id, timestamp, embedding_path "
                    "FROM appearances ORDER BY timestamp",
                )
            else:
                cur = self._conn.execute(
                    "SELECT appearance_id, cluster_id, timestamp, embedding_path "
                    "FROM appearances WHERE cluster_id=? ORDER BY timestamp",
                    (cluster_id,),
                )
            return cur.fetchall()

    def checkpoint(self) -> None:
        """手动 WAL checkpoint (合并 WAL 到主 DB)。"""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self._conn.commit()

    def db_size_bytes(self) -> int:
        """DB 文件总大小 (含 WAL/SHM), 用于监控。"""
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = self.db_path + suffix
            try:
                total += os.path.getsize(p)
            except OSError:
                pass
        return total

    def export_clusters_json(self) -> str:
        """导出 cluster 列表为 JSON。"""
        import json
        clusters = self.list_clusters(order_by="cluster_id ASC", limit=100000)
        return json.dumps([c.to_dict() for c in clusters], indent=2, ensure_ascii=False)

    def export_clusters_csv(self) -> str:
        """导出 cluster 列表为 CSV (header + rows)。"""
        import csv as csv_lib
        import io
        buf = io.StringIO()
        w = csv_lib.writer(buf)
        w.writerow(["cluster_id", "name", "label", "first_seen", "last_seen",
                    "count", "best_emb_path", "best_face_path", "model_version"])
        for c in self.list_clusters(order_by="cluster_id ASC", limit=100000):
            w.writerow([
                c.cluster_id, c.name or "", c.label or "",
                f"{c.first_seen:.6f}", f"{c.last_seen:.6f}",
                c.count, c.best_emb_path or "", c.best_face_path or "",
                c.model_version or "",
            ])
        return buf.getvalue()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            except Exception:
                pass
            self._conn.close()