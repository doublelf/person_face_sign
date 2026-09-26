"""M2 验收程序: 行人检测 → 跟踪 → 人脸检测 → 5点对齐 → 112×112 输出。

用法:
    python apps/m2_track_face.py --source file:/path/to/video.mp4 --show
    python apps/m2_track_face.py --source file:/path/to/video.mp4 --save data/m2_output.mp4
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
from src.detectors.hailo_runner import HailoMultiRunner
from src.tracker.bytetrack import IoUTracker

logger = logging.getLogger("m2_track_face")

DEFAULT_YOLO_HEF = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"
DEFAULT_SCRFD_HEF = "/usr/local/hailo/resources/models/hailo8/scrfd_10g.hef"


def parse_source(uri: str, out_queue: queue.Queue, width: int, height: int, fps: int, loop: bool):
    if uri.startswith("file:"):
        return VideoFileSource(
            path=uri[len("file:"):], out_queue=out_queue,
            width=width, height=height, fps_hint=fps,
            use_realtime=False, loop=loop,
        )
    if uri.startswith("usb:"):
        return USBCameraSource(device=uri[len("usb:"):], out_queue=out_queue,
                               width=width, height=height, fps=fps)
    if uri.startswith("pi:"):
        return PiCameraSource(out_queue=out_queue, camera_id=int(uri[len("pi:"):]),
                              width=width, height=height, fps=fps)
    raise ValueError(f"Unknown source: {uri}")


def color_for_track(tid: int) -> tuple:
    np.random.seed(tid)
    return tuple(int(x) for x in np.random.randint(60, 255, size=3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--yolo-hef", default=DEFAULT_YOLO_HEF)
    ap.add_argument("--scrfd-hef", default=DEFAULT_SCRFD_HEF)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--yolo-score", type=float, default=0.40)
    ap.add_argument("--scrfd-score", type=float, default=0.40)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--save-faces", default="", help="保存对齐后 112×112 人脸的目录")
    ap.add_argument("--log-fps-interval", type=float, default=2.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.save_faces:
        Path(args.save_faces).mkdir(parents=True, exist_ok=True)

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
    total_aligned = 0
    yolo_ms_sum = 0.0
    scrfd_ms_sum = 0.0
    src.start()
    src_min_wait_s = 2.0
    t_start = time.perf_counter()
    t_last_log = t_start
    t_last_frame = t_start
    fps_window = []
    fps_window_max = 30

    try:
        with HailoMultiRunner() as mr:
            mr.add("yolov8_person", args.yolo_hef, input_format_uint8=True, output_format_uint8=False)
            mr.add("scrfd_face", args.scrfd_hef, input_format_uint8=True, output_format_uint8=False)
            with YoloV8PersonDetector(shared_runner=mr, shared_model_name="yolov8_person",
                                       score_thresh=args.yolo_score) as yolo, \
                 SCRFDFaceDetector(shared_runner=mr, shared_model_name="scrfd_face",
                                   score_thresh=args.scrfd_score) as scrfd:
                logger.info("Detectors ready, entering main loop")
            while True:
                if args.max_frames and total_frames >= args.max_frames:
                    break
                if not src.is_running and (time.perf_counter() - t_start) > src_min_wait_s:
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

                # 1) YOLOv8 person
                persons, yolo_ms = yolo.detect_with_timing(frame)
                yolo_ms_sum += yolo_ms
                dets = [(d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.score) for d in persons]

                # 2) Tracker
                sample_tracks = tracker.update(dets)

                # 3) SCRFD on each track's crop
                faces_for_draw = []
                aligned_count_this_frame = 0
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
                    faces, scrfd_ms = scrfd.detect_with_timing(crop)
                    scrfd_ms_sum += scrfd_ms
                    # 关键: 坐标从 crop 空间偏移回原图空间
                    for f in faces:
                        fx1, fy1, fx2, fy2 = f.bbox
                        f.bbox = (fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1)
                        f.landmarks[:, 0] += cx1
                        f.landmarks[:, 1] += cy1
                    if not faces:
                        continue
                    best = max(faces, key=lambda f: f.score)
                    total_faces += 1
                    faces_for_draw.append((t, best))
                    # aligned 用原图坐标系的关键点 (face_align 内部对 112x112 仿射变换)
                    aligned = align_face_5pt(frame, best.landmarks)
                    if args.save_faces:
                        fn = f"{args.save_faces}/t{t.track_id:04d}_f{total_faces:06d}.jpg"
                        cv2.imwrite(fn, aligned)
                    aligned_count_this_frame += 1
                    total_aligned += 1

                # 4) Draw
                vis = frame
                for t in tracker.all_tracks:
                    x1, y1, x2, y2 = map(int, t.bbox)
                    c = color_for_track(t.track_id)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
                    label = f"#{t.track_id} {t.score:.2f} s:{t.sample_count}"
                    cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
                for t, f in faces_for_draw:
                    fx1, fy1, fx2, fy2 = map(int, f.bbox)
                    cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                    for (px, py) in f.landmarks:
                        cv2.circle(vis, (int(px), int(py)), 2, (0, 255, 255), -1)
                    cv2.putText(vis, f"face@{t.track_id} {f.score:.2f}",
                                (fx1, min(frame.shape[0] - 5, fy2 + 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(vis, f"FPS: {cur_fps:.1f} faces: {total_faces} aligned: {total_aligned}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("M2 track+face", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                total_frames += 1
                if t_frame - t_last_log > args.log_fps_interval:
                    logger.info(
                        "frame=%d persons=%d sample_tracks=%d faces=%d aligned=%d "
                        "yolo_ms=%.2f scrfd_ms=%.2f fps=%.1f",
                        total_frames, len(persons), len(sample_tracks),
                        total_faces, total_aligned, yolo_ms, scrfd_ms_sum / max(1, total_faces),
                        cur_fps,
                    )
                    t_last_log = t_frame
    finally:
        elapsed = time.perf_counter() - t_start
        avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
        avg_yolo = yolo_ms_sum / total_frames if total_frames else 0.0
        avg_scrfd = scrfd_ms_sum / max(1, total_faces)
        logger.info(
            "==== M2 summary ====\n"
            "  frames:        %d\n"
            "  faces:         %d\n"
            "  aligned saved: %d\n"
            "  elapsed:       %.2f s\n"
            "  avg FPS:       %.2f\n"
            "  avg yolo ms:   %.2f\n"
            "  avg scrfd ms:  %.2f (per face)",
            total_frames, total_faces, total_aligned, elapsed, avg_fps, avg_yolo, avg_scrfd,
        )
        src.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())