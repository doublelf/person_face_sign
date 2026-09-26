"""M3 验收程序: 行人 → 跟踪 → 人脸 → 对齐 → embedding → 在线聚类 → 持久化。

用法:
    python apps/m3_embed_cluster.py --source file:test.mp4 --db data/gallery.db
    python apps/m3_embed_cluster.py --source file:test.mp4 --show --save data/m3_vis.mp4

输出:
    - SQLite: data/gallery.db (clusters + appearances)
    - Faiss 索引: data/gallery.faiss + .json
    - 可视化: track_id → cluster_id 实时显示
"""
from __future__ import annotations

import argparse
import logging
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.frame_source.video_file import VideoFileSource
from src.frame_source.usb_camera import USBCameraSource
from src.frame_source.pi_camera import PiCameraSource
from src.detectors.yolov8_person import YoloV8PersonDetector
from src.detectors.scrfd_face import SCRFDFaceDetector
from src.detectors.face_align import align_face_5pt
from src.detectors.arcface_embed import ArcFaceEmbedder
from src.detectors.hailo_runner import HailoMultiRunner
from src.tracker.bytetrack import IoUTracker
from src.storage.sqlite_store import SQLiteStore
from src.clustering.faiss_gallery import FaissGallery

logger = logging.getLogger("m3_embed_cluster")

DEFAULT_YOLO_HEF = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"
DEFAULT_SCRFD_HEF = "/usr/local/hailo/resources/models/hailo8/scrfd_10g.hef"
DEFAULT_ARCFACE_HEF = "/usr/local/hailo/resources/models/hailo8/arcface_mobilefacenet.hef"


def parse_source(uri: str, out_queue: queue.Queue, width: int, height: int, fps: int, loop: bool):
    if uri.startswith("file:"):
        return VideoFileSource(path=uri[len("file:"):], out_queue=out_queue,
                               width=width, height=height, fps_hint=fps,
                               use_realtime=False, loop=loop)
    if uri.startswith("usb:"):
        return USBCameraSource(device=uri[len("usb:"):], out_queue=out_queue,
                               width=width, height=height, fps=fps)
    if uri.startswith("pi:"):
        return PiCameraSource(out_queue=out_queue, camera_id=int(uri[len("pi:"):]),
                              width=width, height=height, fps=fps)
    raise ValueError(f"Unknown source: {uri}")


def color_for_cluster(cid: int) -> tuple:
    np.random.seed(cid + 10000)
    return tuple(int(x) for x in np.random.randint(60, 255, size=3))


def quality_ok(face_bbox, landmarks, score: float) -> tuple[bool, str]:
    """检查人脸质量: 尺寸、关键点散布、score。"""
    x1, y1, x2, y2 = face_bbox
    w, h = x2 - x1, y2 - y1
    if w < 30 or h < 30:
        return False, "too_small"
    aspect = w / max(h, 1)
    if not 0.3 < aspect < 2.0:
        return False, "bad_aspect"
    # 关键点散布合理
    if landmarks is not None and len(landmarks) >= 4:
        lms = np.asarray(landmarks[:5])
        eye_y_diff = abs(lms[0, 1] - lms[1, 1])
        mouth_y = (lms[3, 1] + lms[4, 1]) / 2
        if eye_y_diff > h * 0.2:
            return False, "bad_eye_alignment"
        if (mouth_y - lms[0, 1]) < h * 0.1:
            return False, "bad_mouth"
    if score < 0.50:
        return False, "low_score"
    return True, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--yolo-hef", default=DEFAULT_YOLO_HEF)
    ap.add_argument("--scrfd-hef", default=DEFAULT_SCRFD_HEF)
    ap.add_argument("--arcface-hef", default=DEFAULT_ARCFACE_HEF)
    ap.add_argument("--db", default="data/gallery.db")
    ap.add_argument("--faiss", default="data/gallery")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--yolo-score", type=float, default=0.40)
    ap.add_argument("--scrfd-score", type=float, default=0.50)
    ap.add_argument("--cluster-thresh", type=float, default=0.55,
                    help="cosine distance 阈值,低于此归入同 cluster")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--save-faces", default="")
    ap.add_argument("--log-fps-interval", type=float, default=3.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.save_faces:
        Path(args.save_faces).mkdir(parents=True, exist_ok=True)

    store = SQLiteStore(args.db, model_version="m3-v1")
    gallery = FaissGallery(dim=512, distance_threshold=args.cluster_thresh)
    # 加载已有 gallery
    if Path(args.faiss + ".faiss").exists():
        gallery.load(args.faiss)

    frame_q: queue.Queue = queue.Queue(maxsize=4)
    src = parse_source(args.source, frame_q, args.width, args.height, args.fps, args.loop)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    tracker = IoUTracker(max_age=30, min_hits=2, iou_threshold=0.30,
                         sample_interval_frames=60, max_samples_per_track=3)
    total_frames = 0
    total_faces = 0
    total_clusters_new = 0
    total_clusters_matched = 0
    track_to_cluster: dict[int, int] = {}  # track_id -> cluster_id (本进程内)
    cluster_to_color: dict[int, tuple] = {}

    t_start = time.perf_counter()
    t_last_log = t_start
    t_last_frame = t_start
    fps_window = []
    fps_window_max = 30
    src.start()

    try:
        with HailoMultiRunner() as mr:
            mr.add("yolov8_person", args.yolo_hef)
            mr.add("scrfd_face", args.scrfd_hef)
            mr.add("arcface_embed", args.arcface_hef)
            with YoloV8PersonDetector(shared_runner=mr, shared_model_name="yolov8_person",
                                       score_thresh=args.yolo_score) as yolo, \
                 SCRFDFaceDetector(shared_runner=mr, shared_model_name="scrfd_face",
                                   score_thresh=args.scrfd_score) as scrfd, \
                 ArcFaceEmbedder(shared_runner=mr, shared_model_name="arcface_embed") as arcface:
                logger.info("Detectors ready, entering main loop")
                while True:
                    if args.max_frames and total_frames >= args.max_frames:
                        break
                    if not src.is_running and (time.perf_counter() - t_start) > 2.0:
                        break
                    try:
                        _, frame = frame_q.get(timeout=0.5)
                    except queue.Empty:
                        continue
                    t_frame = time.perf_counter()
                    dt_frame = t_frame - t_last_frame
                    if dt_frame > 0:
                        fps_window.append(1.0 / dt_frame)
                        if len(fps_window) > fps_window_max:
                            fps_window.pop(0)
                    t_last_frame = t_frame
                    cur_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

                    persons, _yolo_ms = yolo.detect_with_timing(frame)
                    dets = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score) for d in persons]
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
                        faces, _ = scrfd.detect_with_timing(crop)
                        for f in faces:
                            fx1, fy1, fx2, fy2 = f.bbox
                            f.bbox = (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1)
                            f.landmarks[:, 0] += cx1
                            f.landmarks[:, 1] += cy1
                        if not faces:
                            continue
                        # 先过尺寸过滤 (SCRFD 在小 crop 上常输出 5-20px 噪声框)
                        # 再按 score 选最大
                        candidates = [f for f in faces
                                      if (f.bbox[2] - f.bbox[0]) >= 30
                                      and (f.bbox[3] - f.bbox[1]) >= 30
                                      and f.bbox[0] >= 0 and f.bbox[1] >= 0]
                        if not candidates:
                            continue
                        best = max(candidates, key=lambda f: f.score)
                        ok, why = quality_ok(best.bbox, best.landmarks, best.score)
                        if not ok:
                            continue
                        total_faces += 1
                        aligned = align_face_5pt(frame, best.landmarks)
                        emb = arcface.embed(aligned, normalize=True)
                        # 在线聚类: 找最近 cluster
                        cid_match, dist = gallery.nearest(emb)
                        ts = time.time()
                        if cid_match == -1 or dist > gallery.distance_threshold:
                            new_cid = store.allocate_cluster_id()
                            gallery.add(emb, new_cid)
                            store.upsert_cluster(new_cid, ts, ts, count_delta=1,
                                                 best_face_path=None)
                            store.add_appearance(new_cid, ts, t.track_id, t.bbox,
                                                 best.bbox, None, float(best.score))
                            track_to_cluster[t.track_id] = new_cid
                            total_clusters_new += 1
                        else:
                            store.upsert_cluster(cid_match, ts, ts, count_delta=1)
                            store.add_appearance(cid_match, ts, t.track_id, t.bbox,
                                                 best.bbox, None, float(best.score))
                            track_to_cluster[t.track_id] = cid_match
                            total_clusters_matched += 1
                        if args.save_faces:
                            fn = f"{args.save_faces}/t{t.track_id:04d}_c{track_to_cluster[t.track_id]:04d}_f{total_faces:06d}.jpg"
                            cv2.imwrite(fn, aligned)

                    # 画
                    vis = frame
                    for t in tracker.all_tracks:
                        x1, y1, x2, y2 = map(int, t.bbox)
                        cid = track_to_cluster.get(t.track_id, -1)
                        if cid >= 0:
                            color = cluster_to_color.setdefault(cid, color_for_cluster(cid))
                            label = f"#{t.track_id} c{cid}"
                        else:
                            color = (200, 200, 200)
                            label = f"#{t.track_id}"
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(vis,
                                f"FPS: {cur_fps:.1f} faces: {total_faces} "
                                f"new: {total_clusters_new} match: {total_clusters_matched} "
                                f"clusters: {gallery.n_clusters()}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if writer is not None:
                        writer.write(vis)
                    if args.show:
                        cv2.imshow("M3 embed+cluster", vis)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    total_frames += 1
                    if t_frame - t_last_log > args.log_fps_interval:
                        logger.info(
                            "frame=%d persons=%d sample_tracks=%d faces=%d "
                            "new_c=%d match_c=%d clusters=%d fps=%.1f",
                            total_frames, len(persons), len(sample_tracks),
                            total_faces, total_clusters_new, total_clusters_matched,
                            gallery.n_clusters(), cur_fps,
                        )
                        t_last_log = t_frame
    finally:
        elapsed = time.perf_counter() - t_start
        avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
        logger.info(
            "==== M3 summary ====\n"
            "  frames:        %d\n"
            "  faces:         %d\n"
            "  new clusters:  %d\n"
            "  matched:       %d\n"
            "  total clusters:%d\n"
            "  elapsed:       %.2f s\n"
            "  avg FPS:       %.2f",
            total_frames, total_faces, total_clusters_new, total_clusters_matched,
            gallery.n_clusters(), elapsed, avg_fps,
        )
        # 持久化
        gallery.save(args.faiss)
        store.close()
        src.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())