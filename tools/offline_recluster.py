"""离线重聚类脚本 (单独运行, 不需要在线 pipeline)。

读取 SQLite + Faiss, 跑 Agglomerative 重聚类, 写回 SQLite + Faiss。

用法:
    python tools/offline_recluster.py --db data/gallery.db --faiss data/gallery
                                     --threshold 0.45 --dry-run

⚠️ 重要: 操作会修改 gallery + db, 建议先 --dry-run 看报告
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage.sqlite_store import SQLiteStore
from src.clustering.faiss_gallery import FaissGallery
from src.clustering.offline_recluster import OfflineReclusterer

logger = logging.getLogger("offline_recluster")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gallery.db")
    ap.add_argument("--faiss", default="data/gallery")
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="cosine distance 阈值, 越小越严格")
    ap.add_argument("--linkage", default="average",
                    choices=["average", "complete", "single"])
    ap.add_argument("--dry-run", action="store_true", help="只打印报告, 不修改")
    ap.add_argument("--save-backup", action="store_true",
                    help="修改前先备份 db + faiss")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    store = SQLiteStore(args.db, model_version="recluster-v1")
    gallery = FaissGallery(dim=512)
    if Path(args.faiss + ".faiss").exists():
        gallery.load(args.faiss)

    if len(gallery) == 0:
        logger.warning("Gallery is empty, nothing to do")
        return 0

    if args.save_backup:
        import shutil
        for suffix in ("", ".faiss", ".json"):
            src = args.faiss + (suffix if suffix else "") if suffix else args.db
            if Path(src).exists():
                shutil.copy(src, src + ".bak")
        logger.info("Backup saved")

    # 收集所有向量
    n = gallery._index.ntotal
    vecs = gallery._index.reconstruct_n(0, n).astype(np.float32)
    cur_labels = np.array([gallery._id_to_cluster[i] for i in range(n)])
    logger.info("Loaded %d embeddings, %d unique clusters",
                n, len(set(cur_labels)))

    # 跑重聚类
    rec = OfflineReclusterer(method="agglomerative",
                              distance_threshold=args.threshold,
                              linkage=args.linkage)
    mapping = rec.fit_transform(vecs, cur_labels)

    # 分析 mapping
    old_to_new: dict[int, list[int]] = {}
    for old_cid, new_cid in mapping.items():
        old_to_new.setdefault(int(new_cid), []).append(int(old_cid))

    n_groups = len(old_to_new)
    n_merges = sum(len(olds) - 1 for olds in old_to_new.values() if len(olds) > 1)
    logger.info("=" * 50)
    logger.info("RECLUSTER REPORT")
    logger.info("  threshold:       %.3f", args.threshold)
    logger.info("  linkage:         %s", args.linkage)
    logger.info("  old clusters:    %d", len(set(cur_labels)))
    logger.info("  new clusters:    %d", n_groups)
    logger.info("  merges needed:   %d", n_merges)
    logger.info("=" * 50)

    if args.dry_run:
        logger.info("DRY RUN: no changes applied")
        return 0

    if n_merges == 0:
        logger.info("No merges needed, gallery already consistent")
        return 0

    # 执行 merge
    for new_cid, olds in old_to_new.items():
        if len(olds) <= 1:
            continue
        dest = min(olds)
        for src in olds:
            if src != dest:
                moved = store.merge_clusters(src, dest)
                gallery.merge(src, dest)
                logger.info("Merged cluster %d -> %d (moved %d appearances)",
                            src, dest, moved)

    gallery.save(args.faiss)
    store.checkpoint()
    store.close()
    logger.info("Done. New cluster count: %d", gallery.n_clusters())
    return 0


if __name__ == "__main__":
    sys.exit(main())