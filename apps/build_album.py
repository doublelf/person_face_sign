"""相册构建主程序: 视频文件 → 照片文件夹。

流程:
  video(s) → FrameSampler → 行人+人脸检测 → 对齐 → ArcFace embedding → 聚类 → 保存到 album

用法:
    python apps/build_album.py --video test.mp4 --out data/album
    python apps/build_album.py --video-dir videos/ --out data/album
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.video.sampler import VideoSampler
from src.album.storage import AlbumStorage
from src.album.cluster import AlbumCluster
from src.detectors.yolov8_person import YoloV8PersonDetector
from src.detectors.scrfd_face import SCRFDFaceDetector
from src.detectors.face_align import align_face_5pt
from src.detectors.arcface_embed import ArcFaceEmbedder
from src.detectors.hailo_runner import HailoMultiRunner
from src.tracker.bytetrack import IoUTracker

logger = logging.getLogger("build_album")

DEFAULT_YOLO_HEF = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"
DEFAULT_SCRFD_HEF = "/usr/local/hailo/resources/models/hailo8/scrfd_10g.hef"
DEFAULT_ARCFACE_HEF = "/usr/local/hailo/resources/models/hailo8/arcface_mobilefacenet.hef"

# 质量门槛默认 (M5 修复后)
DEFAULT_SCRFD_SCORE = 0.60
DEFAULT_MIN_FACE_W = 40
DEFAULT_MIN_FACE_H = 40
DEFAULT_MIN_ASPECT = 0.40
DEFAULT_MAX_ASPECT = 1.60
DEFAULT_EYE_Y_RATIO = 0.15

DEFAULT_CLUSTER_THRESH = 0.40
DEFAULT_PHASH_MAX_DIST = 4


def parse_args():
    ap = argparse.ArgumentParser(description="Build face album from videos")
    ap.add_argument("--video", action="append", default=[],
                    help="视频文件,可多次指定或用逗号分隔")
    ap.add_argument("--video-dir", help="视频目录(扫描 *.mp4 *.mov)")
    ap.add_argument("--out", default="data/album")
    ap.add_argument("--yolo-hef", default=DEFAULT_YOLO_HEF)
    ap.add_argument("--scrfd-hef", default=DEFAULT_SCRFD_HEF)
    ap.add_argument("--arcface-hef", default=DEFAULT_ARCFACE_HEF)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--start-offset", type=float, default=0.0)
    ap.add_argument("--end-offset", type=float, default=0.0)
    ap.add_argument("--yolo-score", type=float, default=0.40)
    ap.add_argument("--scrfd-score", type=float, default=DEFAULT_SCRFD_SCORE)
    ap.add_argument("--cluster-thresh", type=float, default=DEFAULT_CLUSTER_THRESH,
                    help="同人 cosine dist 阈值 (M5: 0.40 = sim >= 0.60)")
    ap.add_argument("--dedup-thresh", type=float, default=0.30)
    ap.add_argument("--phash-dist", type=int, default=DEFAULT_PHASH_MAX_DIST)
    ap.add_argument("--min-face-w", type=int, default=DEFAULT_MIN_FACE_W)
    ap.add_argument("--min-face-h", type=int, default=DEFAULT_MIN_FACE_H)
    ap.add_argument("--best-n-embeddings", type=int, default=5)
    ap.add_argument("--skip-dedup", action="store_true")
    ap.add_argument("--log-interval", type=int, default=5)
    # Tracker 参数 (实时模式严, 批处理模式宽)
    ap.add_argument("--tracker-sample-interval", type=int, default=5,
                    help="同一 track 两次采脸的最小间隔(帧); 1 = 每帧都采, 60 = 实时模式")
    ap.add_argument("--tracker-max-samples", type=int, default=20,
                    help="一个 track 最多采几次脸; 短视频调到 999, 实时调到 3")
    ap.add_argument("--tracker-min-hits", type=int, default=2,
                    help="track 确认前需要的命中次数")
    ap.add_argument("--no-tracker", action="store_true",
                    help="完全关掉 tracker, 每帧每个脸都采 (批处理短视频推荐)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    if not args.video and not args.video_dir:
        ap.error("must specify --video or --video-dir")
    return args


def collect_videos(args):
    paths = []
    if vargs := args.video:
        for v in vargs:
            for piece in v.split(","):
                piece = piece.strip()
                if piece:
                    paths.append(piece)
    if args.video_dir:
        vdir = Path(args.video_dir)
        if not vdir.is_dir():
            raise FileNotFoundError(f"video-dir not found: {vdir}")
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.h264", "*.hevc"):
            paths.extend(str(p) for p in sorted(vdir.glob(ext)))
    if not paths:
        raise FileNotFoundError("No video files found")
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def quality_ok(face_bbox, score, min_w, min_h, min_score,
              min_aspect=DEFAULT_MIN_ASPECT, max_aspect=DEFAULT_MAX_ASPECT):
    x1, y1, x2, y2 = face_bbox
    w, h = x2 - x1, y2 - y1
    if w < min_w or h < min_h:
        return False
    aspect = w / max(h, 1)
    if not (min_aspect < aspect < max_aspect):
        return False
    if score < min_score:
        return False
    return True


def eye_alignment_ok(landmarks, face_height, max_ratio=DEFAULT_EYE_Y_RATIO):
    if landmarks is None or len(landmarks) < 5:
        return False
    eye_y_diff = abs(landmarks[0, 1] - landmarks[1, 1])
    return eye_y_diff < face_height * max_ratio


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    video_paths = collect_videos(args)
    logger.info("=" * 60)
    logger.info("BUILD ALBUM")
    logger.info("  videos:        %d files", len(video_paths))
    logger.info("  output:        %s", args.out)
    logger.info("  interval:      %.2f s", args.interval)
    logger.info("  threshold:     %.2f (cosine)", args.cluster_thresh)
    logger.info("=" * 60)
    for v in video_paths:
        logger.info("  - %s", v)

    storage = AlbumStorage(args.out, best_n_embeddings=args.best_n_embeddings)
    cluster = AlbumCluster(threshold=args.cluster_thresh, phash_max_dist=args.phash_dist)
    cluster.bootstrap(storage)
    tracker = IoUTracker(
        max_age=30,
        min_hits=args.tracker_min_hits,
        iou_threshold=0.30,
        sample_interval_frames=(1 if args.no_tracker else args.tracker_sample_interval),
        max_samples_per_track=(999999 if args.no_tracker else args.tracker_max_samples),
    )
    logger.info(
        "Tracker: min_hits=%d sample_interval=%d max_samples=%d%s",
        args.tracker_min_hits,
        1 if args.no_tracker else args.tracker_sample_interval,
        999999 if args.no_tracker else args.tracker_max_samples,
        " (no-tracker mode)" if args.no_tracker else "",
    )

    sampler = VideoSampler(interval_s=args.interval, width=args.width,
                            height=args.height, start_offset_s=args.start_offset,
                            end_offset_s=args.end_offset)
    storage.set_config({
        "interval_s": args.interval,
        "cosine_distance_threshold": args.cluster_thresh,
        "min_face_size": [args.min_face_w, args.min_face_h],
        "min_face_score": args.scrfd_score,
        "best_n_embeddings": args.best_n_embeddings,
        "phash_max_dist": args.phash_dist,
    })

    t_start = time.perf_counter()
    total_frames = 0
    total_faces = 0
    total_persons_created = 0
    total_clusters_matched = 0
    total_deduped = 0
    failed_videos = []

    try:
        with HailoMultiRunner() as mr:
            mr.add("yolov8_person", args.yolo_hef)
            mr.add("scrfd_face", args.scrfd_hef)
            mr.add("arcface_embed", args.arcface_hef)
            with YoloV8PersonDetector(
                shared_runner=mr,
                shared_model_name="yolov8_person",
                score_thresh=args.yolo_score,
            ) as yolo, \
            SCRFDFaceDetector(
                shared_runner=mr,
                shared_model_name="scrfd_face",
                score_thresh=args.scrfd_score,
            ) as scrfd, \
            ArcFaceEmbedder(
                shared_runner=mr,
                shared_model_name="arcface_embed",
            ) as arcface:
                logger.info("Detectors ready")
                for video_path in video_paths:
                    if total_frames >= args.max_frames > 0:
                        break
                    src_name = Path(video_path).stem
                    logger.info("=" * 50)
                    logger.info("Processing: %s", video_path)
                    logger.info("=" * 50)
                    total_duration_s = sampler.probe_duration(video_path)
                    try:
                        for sample in sampler.iter(video_path, src_name):
                            if total_frames >= args.max_frames > 0:
                                break
                            fid, ts, frame, source = sample
                            total_frames += 1
                            try:
                                persons, _ = yolo.detect_with_timing(frame)
                            except Exception as e:
                                logger.exception("YOLO failed at ts=%.2f: %s", ts, e)
                                continue
                            dets = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score)
                                    for d in persons]
                            sample_tracks = tracker.update(dets)

                            for t in sample_tracks:
                                x1, y1, x2, y2 = t.bbox
                                h_f, w_f = frame.shape[:2]
                                pad_w = int((x2 - x1) * 0.1)
                                pad_h = int((y2 - y1) * 0.1)
                                cx1 = max(0, int(x1 - pad_w))
                                cy1 = max(0, int(y1 - pad_h))
                                cx2 = min(w_f, int(x2 + pad_w))
                                cy2 = min(h_f, int(y2 + pad_h))
                                if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                                    continue
                                crop = frame[cy1:cy2, cx1:cx2]
                                try:
                                    faces, _ = scrfd.detect_with_timing(crop)
                                except Exception as e:
                                    logger.exception("SCRFD failed: %s", e)
                                    continue
                                # 偏移坐标到原图
                                for f in faces:
                                    fx1, fy1, fx2, fy2 = f.bbox
                                    f.bbox = (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1)
                                    f.landmarks[:, 0] += cx1
                                    f.landmarks[:, 1] += cy1
                                # 尺寸 + 几何 + score 质量过滤
                                candidates = [f for f in faces
                                              if (f.bbox[2] - f.bbox[0]) >= args.min_face_w
                                              and (f.bbox[3] - f.bbox[1]) >= args.min_face_h
                                              and f.bbox[0] >= 0 and f.bbox[1] >= 0
                                              and quality_ok(
                                                  f.bbox, f.score,
                                                  args.min_face_w, args.min_face_h,
                                                  args.scrfd_score,
                                              )]
                                if not candidates:
                                    continue
                                best = max(candidates, key=lambda f: f.score)
                                face_h = best.bbox[3] - best.bbox[1]
                                if not eye_alignment_ok(best.landmarks, face_h):
                                    logger.debug("Eye alignment fail at ts=%.2f, skipping", ts)
                                    continue
                                aligned = align_face_5pt(frame, best.landmarks)
                                try:
                                    emb = arcface.embed(aligned, normalize=True)
                                except Exception as e:
                                    logger.exception("ArcFace failed: %s", e)
                                    continue
                                total_faces += 1
                                pid, dist, reason = cluster.assign(emb, aligned)
                                if pid == -1 or dist > cluster.threshold:
                                    pid = storage.allocate_person_id()
                                    cluster.add(pid, emb, aligned)
                                    storage.add_person(pid, ts, ts, source)
                                    total_persons_created += 1
                                    logger.info(
                                        "[%s] ts=%.2fs NEW person_%04d (score=%.2f, reason=%s)",
                                        source, ts, pid, best.score, reason,
                                    )
                                else:
                                    logger.debug(
                                        "[%s] ts=%.2fs MATCH person_%04d dist=%.3f reason=%s",
                                        source, ts, pid, dist, reason,
                                    )
                                saved, fpath, dedup_reason = storage.save_face(
                                    person_id=pid,
                                    face_bgr_112=aligned,
                                    embedding=emb,
                                    score=best.score,
                                    timestamp=ts,
                                    source_video=source,
                                    skip_if_similar=not args.skip_dedup,
                                    dedup_threshold=args.dedup_thresh,
                                    phash_max_dist=args.phash_dist,
                                )
                                if not saved:
                                    total_deduped += 1
                                    logger.debug(
                                        "[%s] ts=%.2fs dedup person_%04d (%s)",
                                        source, ts, pid, dedup_reason,
                                    )
                            if total_frames % args.log_interval == 0 or total_frames == 1:
                                logger.info(
                                    "  ... %d frames, %d faces, %d persons, %d dedup",
                                    total_frames, total_faces,
                                    cluster.n_persons(), total_deduped,
                                )
                                # 结构化进度 (供 Web 后台解析)
                                if total_duration_s > 0:
                                    pct = min(100.0, ts / total_duration_s * 100.0)
                                else:
                                    pct = (total_frames / max(total_frames, 1)) * 100.0
                                print(f"[BUILD] PROGRESS {pct:.1f} {total_frames} "
                                      f"{total_faces} {total_persons_created} "
                                      f"{cluster.n_persons()}", flush=True)
                    except Exception as e:
                        logger.exception("Failed on %s: %s", video_path, e)
                        failed_videos.append((video_path, str(e)))
    finally:
        storage.finalize()
        elapsed = time.perf_counter() - t_start
        logger.info("=" * 60)
        logger.info("BUILD COMPLETE")
        logger.info(f"  frames:        {total_frames}")
        logger.info(f"  faces:         {total_faces}")
        logger.info(f"  persons new:   {total_persons_created}")
        logger.info(f"  total persons: {cluster.n_persons()}")
        logger.info(f"  deduped:       {total_deduped}")
        logger.info(f"  elapsed:       {elapsed:.1f} s")
        if elapsed > 0:
            logger.info(f"  fps:           {total_frames / elapsed:.2f}")
        if failed_videos:
            logger.warning(f"  failed videos: {len(failed_videos)}")
            for v, e in failed_videos:
                logger.warning(f"    {v}: {e}")
        logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())