"""生成合成测试视频用于 M1 pipeline 验证。

⚠️ 合成视频不含真实人物,YOLOv8 COCO 模型检测不到 person。
本工具只用于验证: 解码 → 推理 → 可视化 → FPS 测量 整条链路。
真实精度测试需要用户提供含行人的真实视频。

用法:
    python tools/gen_test_video.py --output data/test_pattern.mp4 --duration 30
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def make_pattern_frame(idx: int, total: int, w: int, h: int) -> np.ndarray:
    """生成一帧带移动几何图形的测试图样。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradient background
    for y in range(h):
        ratio = y / max(h - 1, 1)
        img[y, :] = (int(40 + 60 * ratio), int(40 + 60 * (1 - ratio)), int(80 + 100 * ratio))

    # Moving rectangles (simulates persons)
    n_objs = 6
    phase = idx / max(total, 1) * 2 * math.pi
    for i in range(n_objs):
        ang = 2 * math.pi * i / n_objs + phase
        cx = int(w / 2 + (w / 3) * math.cos(ang))
        cy = int(h / 2 + (h / 3) * math.sin(ang))
        bw_ = int(60 + 40 * math.sin(phase * 2 + i))
        bh_ = int(140 + 60 * math.cos(phase * 2 + i))
        color = ((i * 40) % 256, (i * 80 + 100) % 256, (i * 120 + 150) % 256)
        cv2.rectangle(img, (cx - bw_ // 2, cy - bh_ // 2), (cx + bw_ // 2, cy + bh_ // 2), color, -1)
        # 头部小圆
        cv2.circle(img, (cx, cy - bh_ // 2 - 18), 18, color, -1)

    # Time counter
    sec = idx / 25.0
    cv2.putText(img, f"FRAME {idx:06d}  T={sec:.2f}s", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Top-left banner
    cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(img, "SYNTHETIC TEST PATTERN (not real persons)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", "-o", required=True, help="输出 mp4 路径")
    ap.add_argument("--duration", type=float, default=30.0, help="时长(秒)")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(args.duration * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"ERROR: Failed to open writer: {out_path}", file=sys.stderr)
        return 1

    for i in range(total_frames):
        writer.write(make_pattern_frame(i, total_frames, args.width, args.height))
        if i % 25 == 0:
            print(f"\rWrote frame {i}/{total_frames}", end="", flush=True)
    writer.release()
    print(f"\nDone: {out_path} ({total_frames} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())