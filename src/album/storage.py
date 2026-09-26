"""相册文件系统管理 (AlbumStorage)。

职责:
- 创建 person_<id>/ 目录结构
- 写对齐人脸图片
- 更新 representative.jpg
- 维护 embeddings.npz (best N)
- 写 info.json (单个人元数据)
- 写 manifest.json (总体统计) + thumbnails/

设计:
- 单 album 文件夹,所有视频合并
- person ID 全局递增
- 续跑 (append) 模式支持
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def safe_filename(name: str) -> str:
    """替换不合法文件名字符。"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


class AlbumStorage:
    """相册文件系统管理。"""

    def __init__(
        self,
        album_dir: str,
        best_n_embeddings: int = 5,
    ):
        self.album_dir = Path(album_dir)
        self.best_n = best_n_embeddings
        self.thumbnails_dir = self.album_dir / "thumbnails"
        self.album_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        # 加载现有 manifest (续跑用)
        self.manifest_path = self.album_dir / "manifest.json"
        self.manifest = self._load_or_create_manifest()
        # 下一个 person_id
        self._next_person_id = self._compute_next_person_id()
        logger.info(
            "AlbumStorage at %s: %d existing persons, next_id=%d",
            self.album_dir, len(self.manifest.get("persons", [])),
            self._next_person_id,
        )

    def _load_or_create_manifest(self) -> dict:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    return json.load(f)
            except Exception:
                logger.exception("manifest.json corrupt, recre")
        return {
            "version": "1.0",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_persons": 0,
            "total_images": 0,
            "source_videos": [],
            "config": {},
            "persons": [],  # [{id, count, first_ts, last_ts, source_videos, thumbnail}]
        }

    def _compute_next_person_id(self) -> int:
        ids = [p["id"] for p in self.manifest.get("persons", [])]
        return (max(ids) + 1) if ids else 1

    def allocate_person_id(self) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        return pid

    def person_dir(self, person_id: int) -> Path:
        d = self.album_dir / f"person_{person_id:04d}"
        d.mkdir(exist_ok=True)
        return d

    def has_person(self, person_id: int) -> bool:
        return any(p["id"] == person_id for p in self.manifest.get("persons", []))

    def reload_manifest(self) -> None:
        """重新从磁盘读 manifest (build_runner 后调用)。"""
        self.manifest = self._load_or_create_manifest()
        self._next_person_id = self._compute_next_person_id()

    def get_person_ids(self) -> List[int]:
        return [p["id"] for p in self.manifest.get("persons", [])]

    def person_info(self, person_id: int) -> Optional[dict]:
        for p in self.manifest.get("persons", []):
            if p["id"] == person_id:
                return p
        return None

    def list_persons(self, order_by: str = "count DESC",
                     limit: int = 100) -> List[dict]:
        """列出所有 person (dict 形式)。

        order_by: "count DESC" / "cluster_id ASC" / "cluster_id DESC"
        """
        persons = self.manifest.get("persons", [])
        col = order_by.split()[0]
        desc = "DESC" in order_by.upper()
        persons = sorted(persons, key=lambda p: p.get(col, 0), reverse=desc)
        return persons[:limit]

    def list_clusters(self, order_by: str = "count DESC", limit: int = 100):
        """列出所有 person (兼容老 API: 返回 dict list)。"""
        return self.list_persons(order_by=order_by, limit=limit)

    def save_face(
        self,
        person_id: int,
        face_bgr_112: np.ndarray,
        embedding: np.ndarray,
        score: float,
        timestamp: float,
        source_video: str,
        skip_if_similar: bool = True,
        dedup_threshold: float = 0.30,
        phash_max_dist: int = 4,
    ) -> Tuple[bool, str, str]:
        """保存一张人脸 + 更新 metadata。

        Returns: (saved, path, dedup_reason)
            saved=True 表示真的写了盘, path 是图文件路径
            saved=False 表示被去重跳过, dedup_reason 说明原因
        """
        from .dedup import phash_8x8, phash_distance
        pdir = self.person_dir(person_id)
        info_path = pdir / "info.json"
        emb_path = pdir / "embeddings.npz"
        rep_path = pdir / "representative.jpg"
        info = self._load_person_info(person_id)
        # 去重 1: pHash 像素级比对 (针对已有图)
        if skip_if_similar and info["count"] > 0:
            new_hash = phash_8x8(face_bgr_112)
            existing_hashes = []
            for fn in sorted(pdir.glob("img_*.jpg"))[-5:]:
                try:
                    img = cv2.imread(str(fn))
                    if img is not None:
                        existing_hashes.append((fn, phash_8x8(img)))
                except Exception:
                    pass
            for fn, eh in existing_hashes:
                d = phash_distance(new_hash, eh)
                if d <= phash_max_dist:
                    logger.debug("pHash dedup: person %d skip %s dist=%d",
                                 person_id, fn.name, d)
                    return False, "", f"phash_dist={d}"
        # 去重 2: embedding 极相似 (cosine > 1-dedup_threshold)
        if skip_if_similar and info["count"] > 0:
            existing_emb_path = pdir / "embeddings.npz"
            if existing_emb_path.exists():
                try:
                    d = np.load(existing_emb_path)
                    existing_embs = d["embeddings"]
                    if len(existing_embs) > 0:
                        cos = float(existing_embs @ embedding)
                        if cos > 1.0 - dedup_threshold:
                            logger.debug("Embedding dedup: person %d skip ts=%.2f cos=%.3f",
                                         person_id, timestamp, cos)
                            return False, "", f"emb_cos={cos:.3f}"
                except Exception:
                    pass
        # 命名: 4 位 seq (从 count+1)
        seq = info["count"] + 1
        fname = f"img_{seq:04d}_ts_{timestamp:.2f}s_score_{score:.2f}.jpg"
        fpath = pdir / fname
        cv2.imwrite(str(fpath), face_bgr_112)
        # 更新 metadata
        info["count"] += 1
        info["last_ts"] = max(info["last_ts"], timestamp)
        info["first_ts"] = min(info["first_ts"], timestamp)
        if source_video not in info["source_videos"]:
            info["source_videos"].append(source_video)
        # 更新 representative (更高分)
        if score >= info.get("best_score", 0.0):
            info["best_score"] = score
            cv2.imwrite(str(rep_path), face_bgr_112)
            shutil.copy2(str(rep_path),
                         str(self.thumbnails_dir / f"person_{person_id:04d}.jpg"))
        # 更新 embeddings (保持 best N, 按 score 排序)
        embs, scores_list, timestamps_list = [], [], []
        if emb_path.exists():
            try:
                d = np.load(emb_path)
                embs = d["embeddings"].tolist()
                scores_list = d["scores"].tolist()
                timestamps_list = d["timestamps"].tolist()
            except Exception:
                pass
        embs.append(embedding)
        scores_list.append(score)
        timestamps_list.append(timestamp)
        order = np.argsort(-np.array(scores_list))[: self.best_n]
        embs = [embs[i] for i in order]
        scores_list = [scores_list[i] for i in order]
        timestamps_list = [timestamps_list[i] for i in order]
        np.savez_compressed(
            emb_path,
            embeddings=np.stack(embs).astype(np.float32),
            scores=np.array(scores_list, dtype=np.float32),
            timestamps=np.array(timestamps_list, dtype=np.float32),
        )
        self._save_person_info(person_id, info)
        return True, str(fpath), "saved"

    def add_person(
        self,
        person_id: int,
        first_ts: float,
        last_ts: float,
        source_video: str,
        thumbnail_path: Optional[str] = None,
    ) -> None:
        """新建 person 元数据到 manifest。"""
        thumb_rel = f"thumbnails/person_{person_id:04d}.jpg"
        entry = {
            "id": person_id,
            "count": 0,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "best_score": 0.0,
            "source_videos": [source_video],
            "thumbnail": thumb_rel,
        }
        self.manifest["persons"].append(entry)
        # 同步更新 summary
        self._refresh_summary()
        self._save_manifest()

    def update_person_from_info(self, person_id: int) -> None:
        """从 info.json 拉最新数据到 manifest。"""
        info = self._load_person_info(person_id)
        for p in self.manifest["persons"]:
            if p["id"] == person_id:
                p["count"] = info["count"]
                p["first_ts"] = info["first_ts"]
                p["last_ts"] = info["last_ts"]
                p["best_score"] = info.get("best_score", 0.0)
                for v in info["source_videos"]:
                    if v not in p["source_videos"]:
                        p["source_videos"].append(v)
                break
        self._refresh_summary()
        self._save_manifest()

    def _refresh_summary(self):
        self.manifest["total_persons"] = len(self.manifest["persons"])
        self.manifest["total_images"] = sum(p["count"] for p in self.manifest["persons"])
        # 更新 source_videos
        all_videos = set()
        for p in self.manifest["persons"]:
            all_videos.update(p["source_videos"])
        self.manifest["source_videos"] = sorted(all_videos)
        self.manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _person_info_path(self, person_id: int) -> Path:
        return self.person_dir(person_id) / "info.json"

    def _load_person_info(self, person_id: int) -> dict:
        """加载或初始化 person 的 info.json。"""
        p = self._person_info_path(person_id)
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                pass
        # 初始化
        return {
            "count": 0,
            "first_ts": float("inf"),
            "last_ts": -float("inf"),
            "best_score": 0.0,
            "source_videos": [],
        }

    def _save_person_info(self, person_id: int, info: dict):
        with open(self._person_info_path(person_id), "w") as f:
            json.dump(info, f, indent=2)

    def set_config(self, config: dict) -> None:
        self.manifest["config"] = config
        self._save_manifest()

    def finalize(self) -> None:
        """build_album 结束时调用, 写最终 manifest。"""
        # 从每个 person 的 info 拉最新
        for p in list(self.manifest["persons"]):
            self.update_person_from_info(p["id"])
        self._refresh_summary()
        self._save_manifest()
        logger.info("Album finalized: %d persons, %d images",
                    self.manifest["total_persons"],
                    self.manifest["total_images"])