"""极简 Web 服务器: 浏览 album + 静态文件。

设计:
- 一个进程,一个端口 (默认 8080, 0.0.0.0)
- /                 → 静态 HTML (apps/static/index.html)
- /static/*         → 静态文件 (CSS, JS)
- /manifest.json    → album 的 manifest
- /thumb/NNNN.jpg   → thumbnail (走静态目录也可, 这里为兼容旧路径)
- /album/person_NNNN/<filename> → album 内文件
- /album/person_NNNN/images.json → 该 person 的文件名列表

用法:
    python apps/web_browser.py                    # 默认端口 + album=data/album
    python apps/web_browser.py --port 8080 --album data/album

不开 build/upload 功能 (改用 apps/build_album.py CLI)。
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import sys
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("web_browser")

STATIC_DIR = Path(__file__).resolve().parent / "static"


# 防止缓存: 静态文件用弱 ETag (基于 mtime + size)
def _make_etag(path: Path) -> str:
    try:
        st = path.stat()
        return f'W/"{(st.st_mtime_ns ^ st.st_size):x}"'
    except OSError:
        return 'W/"0"'


class AlbumHandler(SimpleHTTPRequestHandler):
    album_dir: Path = Path("data/album")  # 类属性, 由 main  注入

    def log_message(self, fmt, *args):
        # 静默默认 access log, 但保留 server error
        msg = fmt % args
        if " 5" in msg or " 4" in msg[:5]:
            logger.warning(msg)
        else:
            logger.info(msg)

    def end_headers(self):
        # CORS for local file access
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = path + "/index.html" if os.path.exists(path + "/index.html") else None
        if not path or not os.path.exists(path):
            self.send_error(404, "Not Found")
            return None
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        try:
            etag = _make_etag(Path(path))
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.end_headers()
            return path
        except OSError:
            self.send_error(404, "Not Found")
            return None

    def do_GET(self):
        # 拦截特殊路径
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            return self._serve_file(STATIC_DIR / "index.html", "text/html")
        if path == "/static/" or path == "/static":
            return self._redirect("/")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            return self._serve_file(STATIC_DIR / rel)
        if path == "/manifest.json":
            return self._serve_manifest()
        if path.startswith("/thumb/"):
            return self._serve_thumb(path[len("/thumb/"):])
        if path.startswith("/album/"):
            return self._serve_album(path[len("/album/"):])
        # favicon
        if path == "/favicon.ico":
            return self._serve_favicon()
        # 其它
        return self._serve_file(STATIC_DIR / "index.html", "text/html")

    def _serve_file(self, fpath: Path, default_ctype: str | None = None):
        if not fpath.exists() or not fpath.is_file():
            self.send_error(404, "Not Found")
            return
        ctype = mimetypes.guess_type(str(fpath))[0] or default_ctype or "application/octet-stream"
        try:
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("ETag", _make_etag(fpath))
            self.end_headers()
            self.wfile.write(data)
        except OSError as e:
            self.send_error(500, str(e))

    def _redirect(self, to: str):
        self.send_response(302)
        self.send_header("Location", to)
        self.end_headers()

    def _serve_manifest(self):
        mp = self.album_dir / "manifest.json"
        if not mp.exists():
            data = {"total_persons": 0, "total_images": 0,
                    "source_videos": [], "persons": [], "config": {},
                    "created_at": "", "updated_at": ""}
        else:
            with open(mp) as f:
                data = json.load(f)
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_thumb(self, name: str):
        # name: "0001.jpg"
        if not re.match(r"^\d{4}\.jpg$", name):
            self.send_error(400, "bad thumb name")
            return
        p = self.album_dir / "thumbnails" / name
        if not p.exists():
            # 退回 person_NNNN/representative.jpg
            pid = name.split(".")[0]
            p = self.album_dir / f"person_{pid}" / "representative.jpg"
        if not p.exists():
            self.send_error(404, "no thumb")
            return
        self._serve_file(p)

    def _serve_album(self, rel: str):
        # 安全: 不允许 ..
        if ".." in rel or rel.startswith("/"):
            self.send_error(400, "bad path")
            return
        # 特殊: person_NNNN/images.json → 动态生成文件名列表
        m = re.match(r"^person_(\d+)/images\.json$", rel)
        if m:
            return self._serve_images_json(m.group(1))
        p = self.album_dir / rel
        if p.is_dir():
            self.send_error(404, "dir listing not allowed")
            return
        if not p.exists():
            self.send_error(404, "Not Found")
            return
        self._serve_file(p)

    def _serve_images_json(self, pid: str):
        pdir = self.album_dir / f"person_{pid}"
        if not pdir.is_dir():
            self.send_error(404, "no such person")
            return
        images = sorted([f.name for f in pdir.glob("img_*.jpg")])
        body = json.dumps({"person_id": int(pid), "images": images},
                           ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_favicon(self):
        # 返回空 faviconicon (透明 PNG, 1x1)
        import base64
        png_1x1 = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_1x1)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(png_1x1)


def main():
    ap = argparse.ArgumentParser(description="Person Face Album Web Browser")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--album", default="data/album")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    album_dir = Path(args.album).resolve()
    if not album_dir.exists():
        logger.warning("Album dir does not exist: %s (UI will show empty state)",
                       album_dir)
    AlbumHandler.album_dir = album_dir

    server = HTTPServer((args.host, args.port), AlbumHandler)
    logger.info("=" * 60)
    logger.info("Person Face Album Browser (静态 http.server)")
    logger.info("  album:   %s", album_dir)
    logger.info("  static:  %s", STATIC_DIR)
    logger.info("  listen:  http://%s:%d/", args.host, args.port)
    logger.info("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()