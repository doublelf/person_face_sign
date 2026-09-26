"""FastAPI 后台标注接口 + 简易 HTML 页面。

功能:
- GET  /api/health         - 健康检查
- GET  /api/metrics        - Prometheus 格式
- GET  /api/clusters       - cluster 列表 (按 count 倒序)
- GET  /api/clusters/{id}  - cluster 详情 + 最近 appearances
- POST /api/clusters/{id}/name  - 给 cluster 命名
- POST /api/clusters/{id}/label - 给 cluster 分类
- POST /api/clusters/merge - 合并两个 cluster (M5 用)
- GET  /                  - 简易 HTML 管理界面

启动方式:
    uvicorn src.api.admin:app --host 0.0.0.0 --port 8080

或:
    from src.api.admin import start_api_server
    threading.Thread(target=start_api_server, daemon=True).start()
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Body
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    Body = None  # type: ignore
    JSONResponse = None  # type: ignore
    HTMLResponse = None  # type: ignore
    uvicorn = None  # type: ignore

from src.utils.metrics import metrics
from src.utils.health import health

# 全局单例: 由 main app 注册 store + gallery 引用
_store = None
_gallery = None


def set_dependencies(store, gallery):
    global _store, _gallery
    _store = store
    _gallery = gallery


if FASTAPI_AVAILABLE:
    app = FastAPI(title="Person Face Sign API", version="0.1")

    @app.get("/api/health")
    async def api_health():
        return {
            "status": "ok" if health.is_alive() else "degraded",
            "uptime_s": time.time() - metrics._started_at,
            "health": health.status(),
        }

    @app.get("/api/metrics")
    async def api_metrics():
        from fastapi import Response
        return Response(content=metrics.render_prometheus(), media_type="text/plain")

    @app.get("/api/clusters")
    async def api_clusters(limit: int = 100, offset: int = 0):
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        clusters = _store.list_clusters(order_by="count DESC", limit=limit)
        return {
            "total": _gallery.n_clusters() if _gallery else 0,
            "limit": limit,
            "offset": offset,
            "clusters": [c.to_dict() for c in clusters],
        }

    @app.get("/api/clusters/{cluster_id}")
    async def api_cluster_detail(cluster_id: int, recent: int = 10):
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        c = _store.get_cluster(cluster_id)
        if c is None:
            raise HTTPException(404, f"cluster {cluster_id} not found")
        # 拉最近 appearances
        with _store._lock:
            cur = _store._conn.execute(
                "SELECT appearance_id, cluster_id, timestamp, track_id, "
                "pedestrian_bbox, face_bbox, embedding_path, quality_score "
                "FROM appearances WHERE cluster_id=? ORDER BY timestamp DESC LIMIT ?",
                (cluster_id, recent),
            )
            apps_rows = cur.fetchall()
        appearances = []
        for row in apps_rows:
            appearances.append({
                "appearance_id": row[0],
                "cluster_id": row[1],
                "timestamp": row[2],
                "track_id": row[3],
                "pedestrian_bbox": json.loads(row[4]) if row[4] else None,
                "face_bbox": json.loads(row[5]) if row[5] else None,
                "embedding_path": row[6],
                "quality_score": row[7],
            })
        return {
            "cluster": c.to_dict(),
            "appearances": appearances,
        }

    @app.post("/api/clusters/{cluster_id}/name")
    async def api_set_name(cluster_id: int, name: str = Body(..., embed=True)):
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        c = _store.get_cluster(cluster_id)
        if c is None:
            raise HTTPException(404, f"cluster {cluster_id} not found")
        _store.set_cluster_name(cluster_id, name)
        return {"cluster_id": cluster_id, "name": name}

    @app.post("/api/clusters/{cluster_id}/label")
    async def api_set_label(cluster_id: int, label: str = Body(..., embed=True)):
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        c = _store.get_cluster(cluster_id)
        if c is None:
            raise HTTPException(404, f"cluster {cluster_id} not found")
        _store.set_cluster_name(cluster_id, c.name or "", label)
        return {"cluster_id": cluster_id, "label": label}

    @app.post("/api/clusters/merge")
    async def api_merge_clusters(src_id: int = Body(..., embed=True),
                                  dst_id: int = Body(..., embed=True)):
        """合并 src cluster 到 dst cluster。

        Body: {"src_id": 1, "dst_id": 2}
        """
        if _store is None or _gallery is None:
            raise HTTPException(503, "Store/gallery not initialized")
        moved = _store.merge_clusters(src_id, dst_id)
        _gallery.merge(src_id, dst_id)
        return {"src_id": src_id, "dst_id": dst_id, "appearances_moved": moved}

    @app.delete("/api/clusters/{cluster_id}")
    async def api_delete_cluster(cluster_id: int):
        if _store is None or _gallery is None:
            raise HTTPException(503, "Store/gallery not initialized")
        removed = _store.delete_cluster(cluster_id)
        _gallery.remove(cluster_id)
        return {"cluster_id": cluster_id, "appearances_removed": removed}

    @app.post("/api/admin/checkpoint")
    async def api_checkpoint():
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        before = _store.db_size_bytes()
        _store.checkpoint()
        after = _store.db_size_bytes()
        return {"before_bytes": before, "after_bytes": after, "freed_bytes": before - after}

    @app.get("/api/export/clusters.json")
    async def api_export_json():
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        from fastapi import Response
        return Response(content=_store.export_clusters_json(),
                       media_type="application/json",
                       headers={"Content-Disposition": "attachment; filename=clusters.json"})

    @app.get("/api/export/clusters.csv")
    async def api_export_csv():
        if _store is None:
            raise HTTPException(503, "Store not initialized")
        from fastapi import Response
        return Response(content=_store.export_clusters_csv(),
                       media_type="text/csv",
                       headers={"Content-Disposition": "attachment; filename=clusters.csv"})

    @app.post("/api/admin/recluster")
    async def api_recluster(distance_threshold: float = Body(0.45, embed=True)):
        """对当前所有 cluster 跑一次离线重聚类 (Agglomerative)。

        重新分配 cluster_id, 不修改已有 cluster 的名称/标签。
        注意: 本操作会改 _gallery 和 _store 里的 cluster 状态, 比较重。
        """
        if _store is None or _gallery is None:
            raise HTTPException(503, "Store/gallery not initialized")
        from src.clustering.offline_recluster import OfflineReclusterer
        # 收集所有 embeddings
        ids_to_emb: dict[int, list] = {}
        with _store._lock:
            cur = _store._conn.execute(
                "SELECT cluster_id, embedding_path FROM appearances "
                "WHERE embedding_path IS NOT NULL ORDER BY timestamp"
            )
            for row in cur:
                pass  # 我们当前不存 embedding 内容到 path, 跳过
        # 当前不存 embedding 实际值, 只能从 gallery 的 repr 向量凑
        # 简化: 直接用 gallery.ntotal 个向量 + cluster_id
        import numpy as np
        try:
            import faiss
            n = _gallery._index.ntotal
            if n == 0:
                return {"message": "no embeddings"}
            vecs = _gallery._index.reconstruct_n(0, n).astype(np.float32)
            # normed (重建回来是 L2-normed 的)
            cur_labels = np.array([_gallery._id_to_cluster[i] for i in range(n)])
        except Exception as e:
            raise HTTPException(500, f"Cannot extract embeddings: {e}")
        rec = OfflineReclusterer(distance_threshold=distance_threshold)
        mapping = rec.fit_transform(vecs, cur_labels)
        # mapping[old_cid] = new_cid (可能多个 old 映射到同一 new)
        # 对每个 mapping 应用: store.merge_clusters(old, dest)
        # 选 dest = mapping[old_cid] 中最小的 cid 作为 "代表"
        old_to_new = {}
        for old_cid, new_cid in mapping.items():
            old_to_new.setdefault(new_cid, []).append(int(old_cid))
        # 把同 new_cid 下的所有 old 合并到最小 cid
        merges_done = 0
        for new_cid, olds in old_to_new.items():
            if len(olds) <= 1:
                continue
            dest = min(olds)
            for src in olds:
                if src != dest:
                    _store.merge_clusters(src, dest)
                    _gallery.merge(src, dest)
                    merges_done += 1
        return {
            "old_clusters": len(_gallery._cluster_to_id),
            "merges_done": merges_done,
            "new_clusters": _gallery.n_clusters(),
            "distance_threshold": distance_threshold,
        }

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML_PAGE

    HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Person Face Sign - Admin</title>
<style>
body { font-family: sans-serif; margin: 2em; background: #f5f5f5; }
h1 { color: #333; }
.cluster-card { background: white; border-radius: 8px; padding: 1em; margin: 0.5em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.cluster-id { font-weight: bold; color: #0066cc; }
.label-tag { background: #ffd700; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; }
input[type=text] { padding: 4px; border: 1px solid #ccc; border-radius: 4px; }
button { background: #0066cc; color: white; border: none; padding: 4px 12px; border-radius: 4px; cursor: pointer; }
.stat { display: inline-block; background: #fff; padding: 0.5em 1em; margin-right: 1em; border-radius: 4px; }
</style>
</head>
<body>
<h1>Person Face Sign - Admin Console</h1>
<div>
<span class="stat">Total Clusters: <span id="total-clusters">?</span></span>
<span class="stat">Health: <span id="health">?</span></span>
<span class="stat">Uptime: <span id="uptime">?</span>s</span>
</div>
<h2>Clusters</h2>
<button onclick="loadClusters()">Refresh</button>
<div id="clusters"></div>

<script>
async function loadClusters() {
    try {
        const r = await fetch('/api/health');
        const h = await r.json();
        document.getElementById('health').textContent = h.status;
        document.getElementById('uptime').textContent = Math.round(h.uptime_s);

        const rc = await fetch('/api/clusters?limit=50');
        const data = await rc.json();
        document.getElementById('total-clusters').textContent = data.total;

        const div = document.getElementById('clusters');
        div.innerHTML = '';
        for (const c of data.clusters) {
            const card = document.createElement.createElement('div');
            card.className = 'cluster-card';
            card.innerHTML = `
                <div>
                    <span class="cluster-id">#${c.cluster_id}</span>
                    <strong>${c.name || '(unnamed)'}</strong>
                    ${c.label ? '<span class="label-tag">' + c.label + '</span>' : ''}
                    <span style="float: right;">count: ${c.count}</span>
                </div>
                <div style="font-size: 0.8em; color: #666;">
                    first: ${new Date(c.first_seen * 1000).toLocaleString()} ·
                    last: ${new Date(c.last_seen * 1000).toLocaleString()}
                </div>
                <div style="margin-top: 0.5em;">
                    <input type="text" id="name-${c.cluster_id}" placeholder="name" value="${c.name || ''}">
                    <button onclick="setName(${c.cluster_id})">Set Name</button>
                    <input type="text" id="label-${c.cluster_id}" placeholder="label" value="${c.label || ''}">
                    <button onclick="setLabel(${c.cluster_id})">Set Label</button>
                </div>
            `;
            div.appendChild(card);
        }
    } catch (e) {
        document.getElementById('clusters').textContent = 'Error: ' + e.message;
    }
}

async function setName(cid) {
    const name = document.getElementById('name-' + cid).value;
    await fetch('/api/clusters/' + cid + '/name', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
    });
    loadClusters();
}

async function setLabel(cid) {
    const label = document.getElementById('label-' + cid).value;
    await fetch('/api/clusters/' + cid + '/label', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({label: label})
    });
    loadClusters();
}

loadClusters();
setInterval(loadClusters, 5000);
</script>
</body>
</html>"""

else:
    app = None
    logger.warning("FastAPI not available, admin API disabled")


def start_api_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    if not FASTAPI_AVAILABLE or app is None:
        logger.warning("Cannot start admin API: FastAPI not installed")
        return
    logger.info("Starting admin API on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")