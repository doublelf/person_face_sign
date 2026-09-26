"""Faiss HNSW 索引 + 在线贪心聚类。

设计:
- 每个 cluster 用 "best embedding" 代表 (高分人脸的 L2-normalized 向量)
- 新 embedding 来了, 用 HNSW top-1 搜索最近 cluster
- cosine distance < threshold → 归入该 cluster
- 否则新建 cluster

Faiss 用 IndexHNSWFlat + L2 距离 (cosine 等 价于 L2-normed 向量的 L2)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaissGallery:
    """在线贪心聚类的向量索引。

    cluster_id 是外部分配的 (从 SQLiteStore.allocate_cluster_id()).
    我们维护 cluster_id -> 向量 的映射, 用 Faiss 做最近邻搜索。
    """

    def __init__(self, dim: int = 512, distance_threshold: float = 0.55,
                 ef_construction: int = 32, M: int = 16):
        """distance_threshold: cosine distance (1 - similarity)
        - 0.55 ≈ similarity 0.45 (宽松, 容易合)
        - 0.40 ≈ similarity 0.60 (中等)
        - 0.30 ≈ similarity 0.70 (严格, 同人判定要求高)
        """
        self.dim = dim
        self.distance_threshold = distance_threshold
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss-cpu not installed")
        self._faiss = faiss
        self._index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = max(64, ef_construction * 2)
        # internal id (faiss 索引) -> cluster_id (业务 id)
        self._id_to_cluster: List[int] = []
        self._cluster_to_id: dict[int, int] = {}

    def add(self, embedding: np.ndarray, cluster_id: int) -> int:
        """加一条向量。embedding 必须是 L2-normed。"""
        assert embedding.shape == (self.dim,), f"embedding shape {embedding.shape} != ({self.dim},)"
        emb = embedding.reshape(1, -1).astype(np.float32)
        # 保证 L2 normed
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        if cluster_id in self._cluster_to_id:
            # 已存在的 cluster: 暂不更新代表向量 (保守策略)
            return self._cluster_to_id[cluster_id]
        idx = self._index.ntotal
        self._index.add(emb)
        self._id_to_cluster.append(cluster_id)
        self._cluster_to_id[cluster_id] = idx
        return idx

    def update_centroid(self, cluster_id: int, embedding: np.ndarray) -> bool:
        """替换 cluster 的代表向量 (新来的 embedding 质量更好时用)。"""
        if cluster_id not in self._cluster_to_id:
            return False
        # HNSW 不支持 in-place update, 重新加一个新的, 老的保留 (但下次 query 仍可能命中)
        # 简化: 直接重新 add + 标记 cluster 的最新 idx
        emb = embedding.reshape(1, -1).astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        new_idx = self._index.ntotal
        self._index.add(emb)
        self._id_to_cluster.append(cluster_id)
        self._cluster_to_id[cluster_id] = new_idx
        return True

    def search(self, embedding: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (distances, cluster_ids)。embedding 必须是 L2-normed。"""
        emb = embedding.reshape(1, -1).astype(np.float32)
        n = np.linalg.norm(emb)
        if n > 1e-9:
            emb = emb / n
        d, idx = self._index.search(emb, k)
        # idx 可能 -1 (空)
        cluster_ids = np.array([
            self._id_to_cluster[i] if 0 <= i < len(self._id_to_cluster) else -1
            for i in idx[0]
        ])
        return d[0], cluster_ids

    def nearest(self, embedding: np.ndarray) -> Tuple[int, float]:
        """返回最近 cluster 的 (cluster_id, cosine_distance)。"""
        if self._index.ntotal == 0:
            return -1, float("inf")
        d, cids = self.search(embedding, k=1)
        if cids[0] == -1:
            return -1, float("inf")
        return int(cids[0]), float(d[0])

    def __len__(self) -> int:
        return self._index.ntotal

    def n_clusters(self) -> int:
        return len(self._cluster_to_id)

    def has_cluster(self, cluster_id: int) -> bool:
        return cluster_id in self._cluster_to_id

    def merge(self, src_id: int, dst_id: int) -> bool:
        """合并 src cluster 到 dst。Faiss 索引保留两条向量 (因为 HNSW 不支持 in-place remove),
        但 cluster_id 映射改为 dst。

        如果 dst 之前没有 embedding, 会复用 src 的代表向量。
        """
        if src_id == dst_id:
            return False
        if src_id not in self._cluster_to_id:
            return False
        # 如果 dst 还没有向量, 把 src 的代表向量重命名为 dst
        if dst_id not in self._cluster_to_id:
            src_idx = self._cluster_to_id[src_id]
            # 改 id_to_cluster 中的 entry
            self._id_to_cluster[src_idx] = dst_id
            self._cluster_to_id[dst_id] = src_idx
            del self._cluster_to_id[src_id]
            return True
        # 双向都有: 保留两者, 删除 src 的映射
        # 注意: Faiss 索引里的 src 向量还在, 但不再被任何 cluster_id 引用
        del self._cluster_to_id[src_id]
        return True

    def remove(self, cluster_id: int) -> bool:
        """从 cluster 映射中删除。索引里的向量保留 (HNSW 不支持 in-place remove)。"""
        if cluster_id not in self._cluster_to_id:
            return False
        del self._cluster_to_id[cluster_id]
        # 可选: 重建索引彻底清理, 但 N 小时不必要
        return True

    def save(self, path: str) -> None:
        """保存 Faiss 索引 + cluster 映射。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, path + ".faiss")
        import json
        with open(path + ".json", "w") as f:
            json.dump({
                "id_to_cluster": self._id_to_cluster,
                "cluster_to_id": {str(k): v for k, v in self._cluster_to_id.items()},
                "distance_threshold": self.distance_threshold,
            }, f)
        logger.info("FaissGallery saved to %s (%d vectors)", path, self._index.ntotal)

    def load(self, path: str) -> bool:
        """从文件恢复。"""
        try:
            self._index = self._faiss.read_index(path + ".faiss")
            import json
            with open(path + ".json") as f:
                data = json.load(f)
            self._id_to_cluster = data["id_to_cluster"]
            self._cluster_to_id = {int(k): v for k, v in data["cluster_to_id"].items()}
            self.distance_threshold = data.get("distance_threshold", self.distance_threshold)
            logger.info("FaissGallery loaded: %d vectors", self._index.ntotal)
            return True
        except Exception as e:
            logger.warning("FaissGallery load failed: %s", e)
            return False