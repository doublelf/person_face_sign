"""离线重聚类 (AgglomerativeClustering)。

适用场景:
- 在线贪心聚类 (FaissGallery) 累积一段时间后, 阈值 / 边界可能不理想
- 管理员手动合并 / 删除 cluster 后, 想重新校准
- 整体使用 sklearn 的层次聚类, 比在线方法更准确

用法:
    reclusterer = OfflineReclusterer(method='agglomerative', distance_threshold=0.45)
    mapping = reclusterer.fit_transform(embeddings, current_cluster_ids)
    # mapping: {old_cluster_id: new_cluster_id}
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class OfflineReclusterer:
    """层次聚类重整 cluster 划分。"""

    def __init__(
        self,
        method: str = "agglomerative",
        distance_threshold: float = 0.45,
        linkage: str = "average",
        min_cluster_size: int = 2,
    ):
        """
        Args:
            method: 目前只支持 'agglomerative'
            distance_threshold: 合并的 cosine distance 阈值 (0-1)
            linkage: 'average' / 'complete' / 'single'
            min_cluster_size: AgglomerativeClustering 不直接支持, 用 noise filtering
        """
        if method != "agglomerative":
            raise ValueError(f"Unsupported method: {method}")
        self.method = method
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size

    def fit_transform(
        self,
        embeddings: np.ndarray,
        current_cluster_ids: Optional[np.ndarray] = None,
    ) -> Dict[int, int]:
        """输入 embeddings (N, D), 输出 {old_index: new_cluster_id}。

        current_cluster_ids 可选: 如果提供 (长度 N), 输出映射按原 cluster 聚合。
        否则输出映射按 embedding 索引。
        """
        from sklearn.cluster import AgglomerativeClustering
        n = len(embeddings)
        if n == 0:
            return {}
        # 距离矩阵 (cosine distance on L2-normed)
        # sklearn 用 'precomputed' + distance 矩阵
        normed = embeddings.copy()
        norms = np.linalg.norm(normed, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        normed = normed / norms
        # 计算 cosine distance 矩阵
        # 注意: N 较大时这个矩阵是 N^2 内存, 默认 N <= 10000
        if n > 10000:
            logger.warning("N=%d > 10000, skipping (memory)", n)
            return {i: ci for i, ci in enumerate(current_cluster_ids or range(n))}
        cos_sim = normed @ normed.T  # (N, N) in [-1, 1]
        cos_dist = np.clip(1.0 - cos_sim, 0.0, 2.0)
        np.fill_diagonal(cos_dist, 0.0)
        # Agglomerative clustering with precomputed distance
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage=self.linkage,
            )
            new_labels = clusterer.fit_predict(cos_dist)
        except Exception as e:
            logger.exception("AgglomerativeClustering failed: %s", e)
            return {i: ci for i, ci in enumerate(current_cluster_ids or range(n))}

        # 构造映射: old_cluster_id -> new_cluster_id
        mapping: Dict[int, int] = {}
        if current_cluster_ids is None:
            for i in range(n):
                mapping[i] = int(new_labels[i])
        else:
            # 每个 old_cluster_id 选其 embedding 的众数 new_label
            from collections import Counter
            old_to_labels: Dict[int, list[int]] = {}
            for i, old_cid in enumerate(current_cluster_ids):
                old_to_labels.setdefault(int(old_cid), []).append(int(new_labels[i]))
            for old_cid, labels in old_to_labels.items():
                if len(labels) == 1:
                    mapping[old_cid] = labels[0]
                else:
                    most_common = Counter(labels).most_common(1)[0][0]
                    mapping[old_cid] = most_common
        n_new = len(set(new_labels))
        logger.info("OfflineReclusterer: %d items -> %d new clusters (threshold=%.3f)",
                    n, n_new, self.distance_threshold)
        return mapping

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - np.dot(a, b))