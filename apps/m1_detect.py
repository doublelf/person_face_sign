"""M1 验收程序: USB/视频 → 行人检测 → 可视化 + FPS。

Usage:
    # 本地视频
    python apps/m1_detect.py --source file:/path/to/video.mp4 --show

    # USB 摄像头
    python apps/m1_detect.py --source usb:/dev/video0 --show

    # Pi Camera
    python apps/m1_detect.py --source pi:0 --show

输出:
    - 实时画面叠加 bbox 与 FPS
    - 结束/中断时打印统计 (avg FPS, total frames, total detects)
"""
from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.frame_source.video_file import VideoFileSource
from src.frame_source.usb_camera import USBCameraSource
from src.frame_source.pi_camera import PiCameraSource
from src.detectors.yolov8_person import YoloV8PersonDetector

logger = logging.getLogger("m1_detect")

DEFAULT_YOLOV8_HEF = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"


def parse_source(uri: str, out_queue: queue.Queue, width: int, height: int, fps: int, loop: bool):
    """根据 uri 前缀创建 FrameSource。"""
    if uri.startswith("file:"):
        return VideoFileSource(
            path=uri[len("file:"):],
            out_queue=out_queue,
            width=width, height=height,
            fps_hint=fps,
            use_realtime=False,
            loop=loop,
        )
    if uri.startswith("usb:"):
        return USBCameraSource(
            device=uri[len("usb:"):],
            out_queue=out_queue,
            width=width, height=height,
            fps=fps,
        )
    if uri.startswith("pi:"):
        idx = int(uri[len("pi:"):])
        return PiCameraSource(
            out_queue=out_queue,
            camera_id=idx,
            width=width, height=height,
            fps=fps,
        )
    raise ValueError(f"Unknown source uri: {uri}")


def draw_detections(frame, detections, fps):
    """画 bbox 和 FPS。"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # FPS overlay
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description="M1: 行人检测 + 可视化")
    ap.add_argument("--source", required=True,
                    help="file:/path.mp4 | usb:/dev/video0 | pi:0")
    ap.add_argument("--hef", default=DEFAULT_YOLOV8_HEF, help="yolov8n HEF 路径")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--score", type=float, default=0.40, help="置信度阈值")
    ap.add_argument("--max-frames", type=int, default=0, help="最大处理帧数,0=无限")
    ap.add_argument("--show", action="store_true", help="显示实时画面 (cv2.imshow)")
    ap.add_argument("--save", default="", help="保存可视化视频到该路径 (.mp4)")
    ap.add_argument("--loop", action="store_true", help="视频文件循环播放")
    ap.add_argument("--log-fps-interval", type=float, default=2.0, help="FPS 日志打印间隔(秒)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    frame_q: queue.Queue = queue.Queue(maxsize=4)
    src = parse_source(args.source, frame_q, args.width, args.height, args.fps, args.loop)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (args.width, args.height))

    logger.info("Starting source: %s", args.source)
    src.start()

    total_frames = 0
    total_detects = 0
    infer_ms_sum = 0.0
    t_start = time.perf_counter()
    t_last_log = t_start
    t_last_frame = t_start
    fps_window: list[float] = []
    fps_window_max = 30
    # Wait at least this long before accepting source EOF (gives ffmpeg time to start producing)
    src_min_wait_s = 2.0

    try:
        with YoloV8PersonDetector(args.hef, score_thresh=args.score) as detector:
            logger.info("Detector ready, entering main loop")
            while True:
                if args.max_frames and total_frames >= args.max_frames:
                    logger.info("Reached max_frames=%d, exiting", args.max_frames)
                    break
                if not src.is_running and (time.perf_counter() - t_start) > src_min_wait_s:
                    logger.info("Source stopped (EOF?) after %.1fs, exiting", time.perf_counter() - t_start)
                    break
                try:
                    fid, frame = frame_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                detections, infer_ms = detector.detect_with_timing(frame)
                infer_ms_sum += infer_ms
                total_detects += len(detections)
                # Compute FPS
                now = time.perf_counter()
                dt = now - t_last_frame
                if dt > 0:
                    fps_window.append(1.0 / dt)
                    if len(fps_window) > fps_window_max:
                        fps_window.pop(0)
                t_last_frame = now
                cur_fps = sum(fps_window) / len(fps_window) if fps_window else 0.0
                # Periodic log
                if now - t_last_log > args.log_fps_interval:
                    logger.info(
                        "frame=%d detects=%d infer_ms=%.2f fps=%.1f queue=%d",
                        total_frames, len(detections), infer_ms, cur_fps, frame_q.qsize(),
                    )
                    t_last_log = now
                # Visualize
                vis = draw_detections(frame, detections, cur_fps)
                if writer is not None:
                    writer.write(vis)
                if args.show:
                    cv2.imshow("M1 detect", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed 'q', exiting")
                        break
                total_frames += 1
    finally:
        elapsed = time.perf_counter() - t_start
        avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
        avg_infer = (infer_ms_sum / total_frames) if total_frames else 0.0
        logger.info(
            "==== M1 summary ====\n"
            "  frames:        %d\n"
            "  detects:       %d\n"
            "  elapsed:       %.2f s\n"
            "  avg FPS:       %.2f\n"
            "  avg infer ms:  %.2f",
            total_frames, total_detects, elapsed, avg_fps, avg_infer,
        )
        src.stop()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())