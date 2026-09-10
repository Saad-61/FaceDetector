"use strict";

// Dynamically refresh favicon to bust domain-level browser cache (e.g. ngrok tunnel domain)
(function enforceFavicon() {
  try {
    const existing = document.querySelectorAll("link[rel*='icon']");
    const ts = Date.now();
    existing.forEach(el => {
      if (el.href && !el.href.startsWith("data:")) {
        el.href = el.href.split("?")[0] + "?t=" + ts;
      }
    });
  } catch (_) {}
})();

// ── DOM refs ────────────────────────────────────────────────────────────────
const viewUpload  = document.getElementById("view-upload");
const viewLoading = document.getElementById("view-loading");
const viewResults = document.getElementById("view-results");

const dropZone    = document.getElementById("drop-zone");
const fileInput   = document.getElementById("file-input");
const dzIdle      = document.getElementById("dz-idle");
const dzOver      = document.getElementById("dz-over");

const loadingLabel = document.getElementById("loading-label");

const resCount     = document.getElementById("res-count");
const resTime      = document.getElementById("res-time");
const resultImg    = document.getElementById("result-img");
const faceList     = document.getElementById("face-list");
const sidebarCount = document.getElementById("sidebar-count");

const btnNew       = document.getElementById("btn-new");
const errorToast   = document.getElementById("error-toast");
const errorText    = document.getElementById("error-text");
const errorDismiss = document.getElementById("error-dismiss");


// ── Drop zone ───────────────────────────────────────────────────────────────

dropZone.addEventListener("click",   () => fileInput.click());
dropZone.addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) go(fileInput.files[0]);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
  dzIdle.classList.add("hidden");
  dzOver.classList.remove("hidden");
});
dropZone.addEventListener("dragleave", e => {
  if (dropZone.contains(e.relatedTarget)) return;
  dropZone.classList.remove("drag-over");
  dzIdle.classList.remove("hidden");
  dzOver.classList.add("hidden");
});
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  dzIdle.classList.remove("hidden");
  dzOver.classList.add("hidden");
  const f = e.dataTransfer.files?.[0];
  if (!f) return;
  if (!f.type.startsWith("image/")) { showError("Only image files are supported."); return; }
  go(f);
});

btnNew.addEventListener("click",   resetUI);
errorDismiss.addEventListener("click", () => errorToast.classList.add("hidden"));


// ── Upload & detect ─────────────────────────────────────────────────────────

async function prepareImage(file) {
  // If file is already small (<= 2MB), upload directly
  if (file.size <= 2 * 1024 * 1024) {
    return file;
  }

  // Downscale large images (e.g. 10-30MB phone photos) to max 2560px.
  // This reduces upload time over ngrok from 20s to < 0.3s without sacrificing detection quality.
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      const MAX = 2560;
      let { width, height } = img;

      if (width <= MAX && height <= MAX && file.size <= 3 * 1024 * 1024) {
        resolve(file);
        return;
      }

      if (width > MAX || height > MAX) {
        if (width > height) {
          height = Math.round((height * MAX) / width);
          width = MAX;
        } else {
          width = Math.round((width * MAX) / height);
          height = MAX;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);

      canvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), { type: "image/jpeg" }));
          } else {
            resolve(file);
          }
        },
        "image/jpeg",
        0.90
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(file);
    };
    img.src = url;
  });
}

async function go(file) {
  showView("loading");
  loadingLabel.textContent = "Preparing image…";

  try {
    const uploadFile = await prepareImage(file);
    loadingLabel.textContent = "Detecting faces…";

    const form = new FormData();
    form.append("image", uploadFile);

    const res  = await fetch("/detect", {
      method: "POST",
      headers: { "ngrok-skip-browser-warning": "true" },
      body: form,
    });

    let data;
    try {
      data = await res.json();
    } catch (_) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    if (!res.ok || data.error) { showError(data.error || `Server error (${res.status})`); return; }

    render(data);
  } catch (e) {
    showError(e.name === "TypeError"
      ? "Cannot reach server. Is the server and ngrok tunnel running?"
      : "Error: " + e.message);
  }
}


// ── Render results ──────────────────────────────────────────────────────────

function render(data) {
  const n = data.face_count;

  resCount.textContent     = `${n} face${n !== 1 ? "s" : ""} detected`;
  resTime.textContent      = `⏱ ${data.processing_ms} ms`;
  sidebarCount.textContent = n;
  resultImg.src            = data.annotated_image;

  faceList.innerHTML = "";

  if (!n) {
    faceList.innerHTML = `<p style="color:var(--text-3);font-size:12px;padding:12px 6px">
      No faces passed the validator.</p>`;
  } else {
    data.faces.forEach((face, i) => buildCard(face, i));
  }

  showView("results");
}

function buildCard(face, idx) {
  const pct = (face.confidence * 100).toFixed(1);
  const [x1, y1, x2, y2] = face.bbox;
  const w = x2 - x1, h = y2 - y1;

  const card = document.createElement("div");
  card.className = "face-card";

  // Thumbnail
  const thumb = document.createElement("img");
  thumb.className = "face-thumb";
  thumb.src = face.crop;
  thumb.alt = `Face ${idx + 1}`;

  // Info
  const info = document.createElement("div");
  info.className = "face-info";
  info.innerHTML = `
    <span class="face-label">Face ${idx + 1}</span>
    <div class="conf-row">
      <div class="conf-track"><div class="conf-fill" style="width:${pct}%"></div></div>
      <span class="conf-pct">${pct}%</span>
    </div>
  `;

  // Download button
  const dl = document.createElement("button");
  dl.className   = "dl-btn";
  dl.title       = `Download Face ${idx + 1}`;
  dl.textContent = "↓";
  dl.addEventListener("click", (e) => {
    e.stopPropagation();
    downloadCrop(face.crop, idx + 1, pct);
  });

  card.appendChild(thumb);
  card.appendChild(info);
  card.appendChild(dl);
  faceList.appendChild(card);
}

function downloadCrop(dataUri, num, pct) {
  const a    = document.createElement("a");
  a.href     = dataUri;
  a.download = `face_${String(num).padStart(3, "0")}_${pct}pct.jpg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}


// ── View helpers ─────────────────────────────────────────────────────────────

function showView(name) {
  viewUpload.classList.add("hidden");
  viewLoading.classList.add("hidden");
  viewResults.classList.add("hidden");
  if (name === "upload")  viewUpload.classList.remove("hidden");
  if (name === "loading") viewLoading.classList.remove("hidden");
  if (name === "results") viewResults.classList.remove("hidden");
}

function showError(msg) {
  errorText.textContent = msg;
  errorToast.classList.remove("hidden");
  showView("upload");
}

function resetUI() {
  fileInput.value  = "";
  faceList.innerHTML = "";
  resultImg.src    = "";
  errorToast.classList.add("hidden");
  showView("upload");
}
