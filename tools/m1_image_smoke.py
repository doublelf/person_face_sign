"""M1 验收辅助: 单张图检测 + 统计。

用法:
    python tools/m1_image_smoke.py <image_path> [--hef /path/yolov8n.hef]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detectors.yolov8_person import YoloV8PersonDetector

DEFAULT_YOLOV8_HEF = "/home/seeed/wrm/reComputer-R20-CV/src/rpi5_hailo8_yolov8/model/yolov8n.hef"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="输入图像路径")
    ap.add_argument("--hef", default=DEFAULT_YOLOV8_HEF)
    ap.add_argument("--score", type=float, default=0.40)
    ap.add_argument("--save", default="", help="可视化结果保存路径")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image: {args.image}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    print(f"Image: {args.image} ({w}x{h})")

    with YoloV8PersonDetector(args.hef, score_thresh=args.score) as det:
        detections, infer_ms = det.detect_with_timing(img)
    print(f"Inference: {infer_ms:.2f} ms, detections: {len(detections)}")
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d.bbox
        print(f"  [{i}] bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) score={d.score:.3f}")

    if args.save:
        vis = img.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {d.score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(args.save, vis)
        print(f"Saved visualization: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())