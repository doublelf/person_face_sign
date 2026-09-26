// Person Face Album - 极简前端
// 同步加载, 一次拿完 manifest, 后续切换不需 fetch
const albumBase = '/album';   // 由 web_browser.py 路由
const thumbBase = '/thumb';   // 由 web_browser.py 路由

let manifest = null;
let currentPersonId = null;

async function loadManifest() {
  const r = await fetch('/manifest.json');
  if (!r.ok) {
    document.getElementById('grid').innerHTML =
      '<div class="empty">无法加载 manifest.json</div>';
    return;
  }
  manifest = await r.json();
  renderStats();
  renderGrid();
  renderBuildTip();
}

function renderStats() {
  if (!manifest) return;
  document.getElementById('stat-persons').textContent = manifest.total_persons ?? 0;
  document.getElementById('stat-images').textContent = manifest.total_images ?? 0;
  document.getElementById('stat-sources').textContent =
    (manifest.source_videos || []).length;
}

function formatBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1024 / 1024).toFixed(2) + ' MB';
}

function renderBuildTip() {
  // 如果 album 还没数据, 提示怎么 build
  const tip = document.getElementById('build-tip');
  if (manifest && manifest.total_persons === 0) {
    tip.innerHTML = '💡 album 为空。从 CLI build 一个:<br>' +
      '<code style="background:#000;padding:2px 6px;border-radius:3px;">' +
      'python apps/build_album.py --video YOUR.mp4 --out data/album</code>';
    tip.style.display = 'block';
  } else {
    tip.style.display = 'none';
  }
}

function renderGrid() {
  const grid = document.getElementById('grid');
  if (!manifest || !manifest.persons || manifest.persons.length === 0) {
    grid.innerHTML = '<div class="card"><div class="empty">还没有人员数据</div></div>';
    return;
  }
  // 按 count DESC 排序
  const persons = [...manifest.persons].sort(
    (a, b) => (b.count || 0) - (a.count || 0)
  );
  grid.innerHTML = persons.map(p => {
    const id = String(p.id).padStart(4, '0');
    const score = (p.best_score ?? 0).toFixed(2);
    const count = p.count ?? 0;
    return `
      <div class="card" data-id="${p.id}">
        <img src="${thumbBase}/${id}.jpg" loading="lazy"
             onerror="this.src='${albumBase}/person_${id}/representative.jpg'">
        <div class="id">person_${id}</div>
        <div class="meta">${count} 张 · score ${score}</div>
      </div>
    `;
  }).join('');
  grid.querySelectorAll('.card').forEach(el => {
    el.onclick = () => openPerson(parseInt(el.dataset.id));
  });
}

async function openPerson(personId) {
  currentPersonId = personId;
  const id = String(personId).padStart(4, '0');
  const p = manifest.persons.find(x => x.id === personId);
  document.getElementById('album-section').style.display = 'none';
  document.getElementById('detail-section').style.display = 'block';

  // 加载该 person 的图列表 (用 manifest 数据 + 文件 glob)
  let images = [];
  try {
    const r = await fetch(`${albumBase}/person_${id}/images.json`);
    if (r.ok) {
      const data = await r.json();
      images = data.images || [];
    }
  } catch (e) { /* ignore */ }

  // 拼详情页
  document.getElementById('detail').innerHTML = `
    <div id="detail-header">
      <img src="${thumbBase}/${id}.jpg"
           onerror="this.src='${albumBase}/person_${id}/representative.jpg'">
      <div>
        <h3>person_${id}</h3>
        <div class="info">
          <div>图片数: <b>${p?.count ?? 0}</b></div>
          <div>最佳 score: <b>${(p?.best_score ?? 0).toFixed(3)}</b></div>
          <div>视频源: ${(p?.source_videos || []).join(', ') || '-'}</div>
          <div>首次: ${(p?.first_ts ?? 0).toFixed(2)}s |
               末次: ${(p?.last_ts ?? 0).toFixed(2)}s</div>
        </div>
      </div>
    </div>
    <div class="grid" id="img-grid"></div>
  `;
  const ig = document.getElementById('img-grid');
  if (images.length === 0) {
    ig.innerHTML = '<div class="card"><div class="empty">无图片</div></div>';
    return;
  }
  ig.innerHTML = images.map(fn => {
    const url = `${albumBase}/person_${id}/${fn}`;
    return `
      <div class="card" data-url="${url}">
        <img src="${url}" loading="lazy">
        <div class="meta" style="font-family:monospace;font-size:11px;">${fn}</div>
      </div>
    `;
  }).join('');
  ig.querySelectorAll('.card').forEach(el => {
    el.onclick = () => openModal(el.dataset.url);
  });
  window.scrollTo(0, 0);
}

function openModal(url) {
  const m = document.getElementById('modal');
  document.getElementById('modal-img').src = url;
  m.showModal();
}

document.addEventListener('click', e => {
  const m = document.getElementById('modal');
  if (e.target === m) m.close();
});

function closeDetail() {
  document.getElementById('detail-section').style.display = 'none';
  document.getElementById('album-section').style.display = 'block';
  currentPersonId = null;
}

document.getElementById('back-btn').onclick = closeDetail;
document.getElementById('refresh').onclick = () => loadManifest().then(renderBuildTip);

// 启动
loadManifest();