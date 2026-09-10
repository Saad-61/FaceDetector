'use strict';

// ── DOM Elements ─────────────────────────────────────────────────────────────
const viewUpload  = document.getElementById('view-upload');
const viewLoading = document.getElementById('view-loading');
const viewResults = document.getElementById('view-results');

const dropZone    = document.getElementById('drop-zone');
const fileInput   = document.getElementById('file-input');
const dzIdle      = document.getElementById('dz-idle');
const dzOver      = document.getElementById('dz-over');

const resCount    = document.getElementById('res-count');
const resTime     = document.getElementById('res-time');
const resultImg   = document.getElementById('result-img');
const faceList    = document.getElementById('face-list');
const sidebarCount= document.getElementById('sidebar-count');
const btnNew      = document.getElementById('btn-new');

const errorToast  = document.getElementById('error-toast');
const errorText   = document.getElementById('error-text');
const errorDismiss= document.getElementById('error-dismiss');

// ── View Switching ────────────────────────────────────────────────────────────
function showView(view) {
  [viewUpload, viewLoading, viewResults].forEach(v => v.classList.add('hidden'));
  view.classList.remove('hidden');
}

// ── Toast Error ───────────────────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorToast.classList.remove('hidden');
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => errorToast.classList.add('hidden'), 5000);
}
errorDismiss.addEventListener('click', () => errorToast.classList.add('hidden'));

// ── Drop Zone Events ──────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dzIdle.classList.add('hidden');
  dzOver.classList.remove('hidden');
  dropZone.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach(ev => {
  dropZone.addEventListener(ev, () => {
    dzIdle.classList.remove('hidden');
    dzOver.classList.add('hidden');
    dropZone.classList.remove('drag-over');
  });
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dzIdle.classList.remove('hidden');
  dzOver.classList.add('hidden');
  dropZone.classList.remove('drag-over');

  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) handleFile(file);
  fileInput.value = '';
});

btnNew.addEventListener('click', () => showView(viewUpload));

// ── File Handling & Upload ───────────────────────────────────────────────────
function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file (JPG, PNG, WEBP, BMP).');
    return;
  }
  upload(file);
}

async function upload(file) {
  showView(viewLoading);

  const fd = new FormData();
  fd.append('image', file);

  try {
    const res = await fetch('/detect', {
      method: 'POST',
      body: fd,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Server error ' + res.status }));
      throw new Error(err.error || 'Server error ' + res.status);
    }

    const data = await res.json();
    renderResults(data);
    showView(viewResults);

  } catch (err) {
    showView(viewUpload);
    showError(err.message || 'Detection failed. Check server connection.');
  }
}

// ── Render Results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const n = data.face_count;
  const t = Math.round(data.processing_ms);

  resCount.textContent = `${n} face${n === 1 ? '' : 's'} detected`;
  resTime.textContent  = `in ${t} ms`;
  sidebarCount.textContent = `(${n})`;

  resultImg.src = data.annotated_image;

  faceList.innerHTML = '';

  if (n === 0) {
    const empty = document.createElement('div');
    empty.className = 'face-empty';
    empty.textContent = 'No faces detected';
    faceList.appendChild(empty);
    return;
  }

  data.faces.forEach((face, idx) => {
    const card = document.createElement('div');
    card.className = 'face-card';

    const thumb = document.createElement('img');
    thumb.className = 'face-thumb';
    thumb.src = face.crop;
    thumb.alt = `Face #${idx + 1}`;

    const info = document.createElement('div');
    info.className = 'face-info';

    const num = document.createElement('span');
    num.className = 'face-num';
    num.textContent = `#${idx + 1}`;

    const score = document.createElement('span');
    score.className = 'face-score';
    const c = Math.round((face.confidence ?? 1) * 100);
    score.textContent = `${c}%`;

    info.appendChild(num);
    info.appendChild(score);
    card.appendChild(thumb);
    card.appendChild(info);
    faceList.appendChild(card);
  });
}
