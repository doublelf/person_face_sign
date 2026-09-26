"""相册内存聚类管理 (AlbumCluster)。

设计:
- 启动时从 AlbumStorage 加载现有 person 的代表向量
- 新 embedding 来, 用 Faiss HNSW 找最近
- cosine distance < threshold → 归入该 person
- 否则 → 分配新 person_id

与 FaissGallery 的区别:
- 本类只用于**批处理** (build_album), 状态在内存, 不持久化
- 不存每张图的向量, 只存代表向量 (best N 由 AlbumStorage 负责)
- 不需要 save/load (重启时从磁盘重建)

M5 修复 (本次):
- 默认 threshold = 0.40 (cosine dist), 即 similarity >= 0.60 算同一人
- 集成 pHash: 同时维护每个 person 的代表 pHash
- assign() 先查 pHash 再查 embedding
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dedup import phash_8x8, phash_distance

logger = logging.getLogger(__name__)


class AlbumCluster:
    """内存聚类 (批处理模式).

    Usage:
        cluster = AlbumCluster(threshold=0.40)
        cluster.bootstrap(storage)  # 加载已有 person
        person_id, dist = cluster.assign(embedding, aligned_face)
    """

    def __init__(
        self,
        threshold: float = 0.40,
        dim: int = 512,
        phash_max_dist: int = 4,
    ):
        """
        Args:
            threshold: cosine distance 阈值 (sim >= 1 - threshold 算同人)
            phash_max_dist: pHash Hamming 距离阈值 (≤ 算同图)
        """
        self.threshold = threshold
        self.dim = dim
        self.phash_max_dist = phash_max_dist
        # 内存索引: cluster_id -> 代表向量 (best score 的 embedding)
        self._repr_vecs: dict[int, np.ndarray] = {}
        # 每个 cluster 的代表 pHash (同图去重)
        self._repr_phash: dict[int, int] = {}
        # 简单 Faiss 索引 (HNSW)
        self._index = None
        self._id_to_cluster: List[int] = []
        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_L2)
            self._index.hnsw.efConstruction = 32
            self._index.hnsw.efSearch = 64
        except ImportError:
            logger.warning("faiss not available, falling back to brute force")
            self._faiss = None

    def bootstrap(self, storage, best_n: int = 5) -> None:
        """从 AlbumStorage 加载所有现有 person 的代表向量 + pHash。

        代表向量: 每个 person 的 embeddings.npz 中 score 最高的那个 (即 npy[0])。
        代表图: representative.jpg → pHash。
        """
        loaded = 0
        for pid in storage.get_person_ids():
            pdir = storage.person_dir(pid)
            # 加载代表向量
            emb_path = pdir / "embeddings.npz"
            if emb_path.exists():
                try:
                    d = np.load(emb_path)
                    embs = d["embeddings"]
                    if len(embs) > 0:
                        self._add_internal(pid, embs[0])
                except Exception as e:
                    logger.warning("Failed to load embeddings for person %d: %s", pid, e)
            # 加载代表图 pHash
            rep_path = pdir / "representative.jpg"
            if rep_path.exists():
                try:
                    import cv2
                    img = cv2.imread(str(rep_path))
                    if img is not None:
                        self._repr_phash[pid] = phash_8x8(img)
                except Exception as e:
                    logger.warning("Failed to load representative for person %d: %s", pid, e)
            loaded += 1
        logger.info("AlbumCluster bootstrapped with %d persons", loaded)

    def _add_internal(self, person_id: int, vec: np.ndarray) -> None:
        self._repr_vecs[person_id] = vec
        if self._index is not None and self._index.ntotal < 50000:
            v = vec.astype(np.float32).reshape(1, -1)
            self._index.add(v)
            self._id_to_cluster.append(person_id)

    def assign(
        self,
        embedding: np.ndarray,
        aligned_face: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, str]:
        """找最近 cluster (pHash 优先, embedding 兜底)。

        Args:
            embedding: 512-d L2-normed 向量
            aligned_face: 112×112 对齐人脸 (用于 pHash)

        Returns:
            (person_id, cosine_dist, reason)
            reason: 'phash' / 'embedding' / 'none'
        """
        # 1. pHash 查重 (跨所有 person)
        if aligned_face is not None:
            new_hash = phash_8x8(aligned_face)
            best_pid = -1
            best_dist = self.phash_max_dist + 1
            for pid, h in self._repr_phash.items():
                d = phash_distance(new_hash, h)
                if d < best_dist:
                    best_dist = d
                    best_pid = pid
            if best_pid >= 0 and best_dist <= self.phash_max_dist:
                # pHash 命中 → 返回 0 distance
                return best_pid, 0.0, "phash"
        # 2. Embedding 最近邻
        pid, dist = self._assign_embedding(embedding)
        if pid >= 0:
            return pid, dist, "embedding"
        return -1, float("inf"), "none"

    def _assign_embedding(self, embedding: np.ndarray) -> Tuple[int, float]:
        emb = embedding.astype(np.float32).reshape(1, -1)
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        if not self._repr_vecs:
            return -1, float("inf")
        # Faiss
        if self._index is not None and self._index.ntotal > 0:
            d, idx = self._index.search(emb, 1)
            d = float(d[0][0])
            ci = int(idx[0][0])
            if 0 <= ci < len(self._id_to_cluster):
                return self._id_to_cluster[ci], d
        # Fallback: brute force
        best_id, best_d = -1, float("inf")
        for pid, vec in self._repr_vecs.items():
            d = float(np.linalg.norm(vec - emb))
            if d < best_d:
                best_id, best_d = pid, d
        return best_id, best_d

    def add(
        self,
        person_id: int,
        embedding: np.ndarray,
        aligned_face: Optional[np.ndarray] = None,
    ) -> None:
        """把新 person 的代表向量 + pHash 加入索引。"""
        emb = embedding.astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        self._add_internal(person_id, emb)
        if aligned_face is not None:
            self._repr_phash[person_id] = phash_8x8(aligned_face)

    def update_phash(self, person_id: int, aligned_face: np.ndarray) -> None:
        """更新 person 的 pHash (e.g., representative 换了更优的图)。"""
        if person_id in self._repr_phash or person_id in self._repr_vecs:
            self._repr_phash[person_id] = phash_8x8(aligned_face)

    def remove(self, person_id: int) -> None:
        self._repr_vecs.pop(person_id, None)
        self._repr_phash.pop(person_id, None)

    def __len__(self) -> int:
        return len(self._repr_vecs)

    def n_persons(self) -> int:
        return len(self._repr_vecs)