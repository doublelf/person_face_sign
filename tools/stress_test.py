"""M4 验收: 长期稳定性压测工具。

用法:
    # 默认: 跑 data/test_people.mp4, loop 60 分钟
    python tools/stress_test.py --duration 3600

    # 跑 1 小时, 定期打印资源占用
    python tools/stress_test.py --duration 3600 --memory-warn-mb 800

测量指标:
- 总运行时间
- 处理帧数
- 平均 FPS
- 检测到的人脸数 / 集群数
- 进程内存变化 (起始 / 当前 / 峰值)
- 健康失败次数
"""
from __future__ import annotations

import argparse
import gc
import logging
import queue
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.frame_source.video_file import VideoFileSource
from src.detectors.yolov8_person import YoloV8PersonDetector
from src.detectors.scrfd_face import SCRFDFaceDetector
from src.detectors.face_align import align_face_5pt
from src.detectors.arcface_embed import ArcFaceEmbedder
from src.detectors.hailo_runner import HailoMultiRunner
from src.tracker.bytetrack import IoUTracker
from src.storage.sqlite_store import SQLiteStore
from src.clustering.faiss_gallery import FaissGallery
from src.utils.health import HealthMonitor
from src.utils.metrics import Metrics, update_resource_metrics

logger = logging.getLogger("stress_test")

DEFAULT_VIDEO = "/home/seeed/person_face_sign/data/test_people.mp4"
DEFAULT_YOLO = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"
DEFAULT_SCRFD = "/usr/local/hailo/resources/models/hailo8/scrfd_10g.hef"
DEFAULT_ARCFACE = "/usr/local/hailo/resources/models/hailo8/arcface_mobilefacenet.hef"


def run_stress_test(args):
    metrics = Metrics()
    health = HealthMonitor(
        max_consecutive_failures=args.max_failures,
        inference_timeout_s=args.inference_timeout,
    )

    proc = psutil.Process()
    mem_start_mb = proc.memory_info().rss / 1024 / 1024
    mem_peak_mb = mem_start_mb
    metrics.set_gauge("memory_mb_start", mem_start_mb)

    logger.info("=" * 60)
    logger.info("STRESS TEST")
    logger.info("  duration: %ds (%.1f min)", args.duration, args.duration / 60)
    logger.info("  video:    %s", args.video)
    logger.info("  loop:     %s", args.loop)
    logger.info("  mem start: %.1f MB", mem_start_mb)
    logger.info("=" * 60)

    store = SQLiteStore(args.db, model_version="stress-v1")
    gallery = FaissGallery(dim=512, distance_threshold=0.55)
    if Path(args.faiss + ".faiss").exists():
        gallery.load(args.faiss)

    frame_q: queue.Queue = queue.Queue(maxsize=4)
    src = VideoFileSource(args.video, frame_q, width=args.width, height=args.height,
                          fps_hint=args.fps, use_realtime=False, loop=args.loop)
    tracker = IoUTracker(max_age=30, min_hits=2, iou_threshold=0.30,
                         sample_interval_frames=60, max_samples_per_track=3)
    track_to_cluster: dict = {}

    t_start = time.perf_counter()
    t_end = t_start + args.duration
    t_last_log = t_start
    total_frames = 0
    total_faces = 0
    total_clusters_new = 0
    crash_count = 0
    last_frame_id = -1
    loop_count = 0

    src.start()
    try:
        with HailoMultiRunner() as mr:
            mr.add("yolov8_person", args.yolo_hef)
            mr.add("scrfd_face", args.scrfd_hef)
            mr.add("arcface_embed", args.arcface_hef)
            with YoloV8PersonDetector(shared_runner=mr, shared_model_name="yolov8_person",
                                       score_thresh=0.40) as yolo, \
                 SCRFDFaceDetector(shared_runner=mr, shared_model_name="scrfd_face",
                                   score_thresh=0.50) as scrfd, \
                 ArcFaceEmbedder(shared_runner=mr, shared_model_name="arcface_embed") as arcface:
                logger.info("Detectors ready, entering loop")
                while time.perf_counter() < t_end:
                    health.heartbeat()
                    try:
                        _, frame = frame_q.get(timeout=2.0)
                    except queue.Empty:
                        if args.loop:
                            loop_count += 1
                            continue
                        else:
                            logger.warning("No frames for 2s, exiting")
                            break
                    # frame_id tracking
                    cur_fid = total_frames
                    last_frame_id = cur_fid
                    # detect
                    try:
                        persons, _ = yolo.detect_with_timing(frame)
                        health.record_success("yolo")
                    except Exception as e:
                        health.record_failure(f"yolo: {e}", "yolo")
                        crash_count += 1
                        continue
                    dets = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score) for d in persons]
                    sample_tracks = tracker.update(dets)

                    for t in sample_tracks:
                        x1, y1, x2, y2 = t.bbox
                        h_f, w_f = frame.shape[:2]
                        pad_w, pad_h = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
                        cx1, cy1 = max(0, int(x1 - pad_w)), max(0, int(y1 - pad_h))
                        cx2, cy2 = min(w_f, int(x2 + pad_w)), min(h_f, int(y2 + pad_h))
                        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                            continue
                        crop = frame[cy1:cy2, cx1:cx2]
                        try:
                            faces, _ = scrfd.detect_with_timing(crop)
                            health.record_success("scrfd")
                        except Exception as e:
                            health.record_failure(f"scrfd: {e}", "scrfd")
                            crash_count += 1
                            continue
                        for f in faces:
                            fx1, fy1, fx2, fy2 = f.bbox
                            f.bbox = (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1)
                            f.landmarks[:, 0] += cx1
                            f.landmarks[:, 1] += cy1
                        cands = [f for f in faces
                                 if (f.bbox[2] - f.bbox[0]) >= 30
                                 and (f.bbox[3] - f.bbox[1]) >= 30
                                 and f.bbox[0] >= 0 and f.bbox[1] >= 0]
                        if not cands:
                            continue
                        best = max(cands, key=lambda f: f.score)
                        if best.score < 0.5:
                            continue
                        aligned = align_face_5pt(frame, best.landmarks)
                        try:
                            emb = arcface.embed(aligned, normalize=True)
                            health.record_success("arcface")
                        except Exception as e:
                            health.record_failure(f"arcface: {e}", "arcface")
                            crash_count += 1
                            continue
                        cid_match, dist = gallery.nearest(emb)
                        ts = time.time()
                        if cid_match == -1 or dist > gallery.distance_threshold:
                            new_cid = store.allocate_cluster_id()
                            gallery.add(emb, new_cid)
                            store.upsert_cluster(new_cid, ts, ts, count_delta=1)
                            store.add_appearance(new_cid, ts, t.track_id, t.bbox, best.bbox,
                                                 None, float(best.score))
                            track_to_cluster[t.track_id] = new_cid
                            total_clusters_new += 1
                        else:
                            store.upsert_cluster(cid_match, ts, ts, count_delta=1)
                            store.add_appearance(cid_match, ts, t.track_id, t.bbox, best.bbox,
                                                 None, float(best.score))
                            track_to_cluster[t.track_id] = cid_match
                        total_faces += 1

                    total_frames += 1
                    # 资源监控
                    mem_now = proc.memory_info().rss / 1024 / 1024
                    mem_peak_mb = max(mem_peak_mb, mem_now)
                    if mem_now > args.memory_kill_mb:
                        logger.error("Memory %d MB exceeds kill threshold, stopping", mem_now)
                        break
                    if mem_now > args.memory_warn_mb:
                        logger.warning("Memory %d MB exceeds warn threshold", mem_now)
                    update_resource_metrics()
                    # Periodic log
                    now = time.perf_counter()
                    if now - t_last_log > args.log_interval:
                        elapsed = now - t_start
                        fps_avg = total_frames / elapsed
                        logger.info(
                            "[%5ds] frames=%d faces=%d clusters=%d fps=%.1f "
                            "mem=%.1fMB (peak %.1f) crashes=%d health_fail=%d "
                            "loops=%d",
                            int(elapsed), total_frames, total_faces, gallery.n_clusters(),
                            fps_avg, mem_now, mem_peak_mb, crash_count,
                            health.status()["total_failures"], loop_count,
                        )
                        t_last_log = now
                        # 周期 gc
                        gc.collect()
    except Exception as e:
        logger.exception("FATAL: stress test crashed")
        crash_count += 1
    finally:
        src.stop()
        gallery.save(args.faiss)
        store.close()
        elapsed = time.perf_counter() - t_start
        fps_avg = total_frames / elapsed if elapsed > 0 else 0
        mem_end_mb = proc.memory_info().rss / 1024 / 1024
        report = {
            "elapsed_s": elapsed,
            "frames": total_frames,
            "faces": total_faces,
            "new_clusters": total_clusters_new,
            "total_clusters": gallery.n_clusters(),
            "avg_fps": fps_avg,
            "mem_start_mb": mem_start_mb,
            "mem_end_mb": mem_end_mb,
            "mem_peak_mb": mem_peak_mb,
            "mem_growth_mb": mem_end_mb - mem_start_mb,
            "crashes": crash_count,
            "health_failures": health.status()["total_failures"],
            "loops": loop_count,
        }
        logger.info("=" * 60)
        logger.info("STRESS TEST REPORT")
        for k, v in report.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)
        return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=600, help="运行时长(秒)")
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--db", default="/home/seeed/person_face_sign/data/stress_test.db")
    ap.add_argument("--faiss", default="/home/seeed/person_face_sign/data/stress_test")
    ap.add_argument("--yolo-hef", default=DEFAULT_YOLO)
    ap.add_argument("--scrfd-hef", default=DEFAULT_SCRFD)
    ap.add_argument("--arcface-hef", default=DEFAULT_ARCFACE)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--max-failures", type=int, default=10)
    ap.add_argument("--inference-timeout", type=float, default=30.0)
    ap.add_argument("--memory-warn-mb", type=float, default=800.0)
    ap.add_argument("--memory-kill-mb", type=float, default=1500.0)
    ap.add_argument("--log-interval", type=float, default=30.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = run_stress_test(args)
    # 退出码: 0 通过, 1 内存超限, 2 崩溃过多
    if report is None:
        return 2
    if report["crashes"] > 50:
        return 2
    if report["mem_peak_mb"] > args.memory_kill_mb * 0.95:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())