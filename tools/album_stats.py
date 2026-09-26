"""查看 album 统计信息。

打印:
- 总人数 / 总图片
- 每个人 (id, count, best_score, source_videos)
- album 目录大小

用法:
    python tools/album_stats.py --album data/album
    python tools/album_stats.py --album data/album --top 5 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", default="data/album")
    ap.add_argument("--top", type=int, default=20,
                    help="显示前 N 个 (按 count 降序)")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    album_dir = Path(args.album)
    manifest_path = album_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}")
        return 1
    with open(manifest_path) as f:
        m = json.load(f)

    # 计算目录总大小
    total_bytes = 0
    file_count = 0
    for root, dirs, files in os.walk(album_dir):
        for fn in files:
            p = os.path.join(root, fn)
            total_bytes += os.path.getsize(p)
            file_count += 1

    if args.json:
        print(json.dumps({
            "total_persons": m.get("total_persons", 0),
            "total_images": m.get("total_images", 0),
            "source_videos": m.get("source_videos", []),
            "config": m.get("config", {}),
            "disk_bytes": total_bytes,
            "file_count": file_count,
        }, indent=2))
        return 0

    print(f"=== Album: {album_dir} ===")
    print(f"  total_persons:  {m.get('total_persons', 0)}")
    print(f"  total_images:   {m.get('total_images', 0)}")
    print(f"  source_videos:  {m.get('source_videos', [])}")
    print(f"  disk:           {total_bytes / 1024:.1f} KB ({file_count} files)")
    print(f"  config:          {json.dumps(m.get('config', {}), ensure_ascii=False)}")
    print()

    # 排序: count DESC
    persons = sorted(m.get("persons", []), key=lambda p: p.get("count", 0), reverse=True)
    print(f"=== Top {args.top} persons (by count) ===")
    print(f"  {'ID':>4}  {'count':>6}  {'best_score':>10}  {'first_ts':>8}  {'last_ts':>8}  {'src'}")
    for p in persons[:args.top]:
        pid = p["id"]
        print(f"  {pid:04d}  {p.get('count', 0):>6}  {p.get('best_score', 0):>10.3f}  "
              f"{p.get('first_ts', 0):>8.2f}  {p.get('last_ts', 0):>8.2f}  "
              f"{','.join(p.get('source_videos', []))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())