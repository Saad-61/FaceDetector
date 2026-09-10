# Face Detection Module — Full Implementation Plan

---

## Project Layout (Final)

```
face_detect_app/
│
├── face_detection/           ← Python detection module (library)
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   ├── detector.py
│   ├── tiling.py
│   ├── validator.py
│   ├── postprocess.py
│   ├── models/
│   │   └── scrfd.py
│   └── utils.py
│
├── app/                      ← Web frontend + API
│   ├── server.py             ← Flask app (API + serves frontend)
│   └── static/
│       ├── index.html        ← Drag-and-drop UI
│       ├── style.css
│       └── main.js
│
├── requirements.txt
└── run.py                    ← Entry point: python run.py
```

---

## ❓ Does the 2-Stage Pipeline Add Overhead?

**Short answer: No. Stage 2 is effectively free.**

Here is the actual time breakdown per image:

| Stage | What runs | Typical time |
|---|---|---|
| Image loading | OpenCV imread / numpy copy | ~5–20 ms |
| Tiling (if large image) | Python slicing, no model | ~1–5 ms |
| **Stage 1 — SCRFD inference** | Neural network, ONNX runtime | **95–99% of total time** |
| **Stage 2 — Validator** | Pure Python math on a tiny list of dicts | **< 1 ms regardless of N** |
| NMS | ~O(N²) on tiny N, numpy | < 0.5 ms |

**Why Stage 2 is free:**
- It operates on the *output* of the model — a short list of bounding box dicts (typically < 100 per image)
- It does only arithmetic: comparisons, additions, divisions
- No GPU, no neural network, no memory allocation
- Even with 200 candidate faces, all 5 layers finish in under 1 ms

**The actual bottleneck is always Stage 1 (SCRFD).** Tiling multiplies Stage 1 proportionally (4 tiles ≈ 4× inference time) — that's the cost of detecting small faces. Stage 2 never contributes meaningfully to latency.

**Realistic total times (CPU, SCRFD-10GF):**

| Image & scenario | Tiling? | Estimated total |
|---|---|---|
| 640×480 — 1 person | No | ~60–100 ms |
| 1920×1080 — group photo | No | ~120–200 ms |
| 1920×1080 — crowd | Yes (~6 tiles) | ~400–700 ms |
| 3840×2160 — crowd | Yes (~24 tiles) | ~1.5–3 s |

With **GPU (CUDA)**, divide all numbers by roughly 8–15×.

> Stage 2 adds < 1 ms in every scenario. It costs nothing.

---

## The Core Tension: Recall vs. Precision

| Goal | What it means | Naive approach | Problem |
|---|---|---|---|
| **High Recall** | Find every face | Lower confidence threshold | More false positives |
| **High Precision** | No false positives | Raise confidence threshold | Real faces get missed |

**Solution — two-stage pipeline:**
- **Stage 1 (SCRFD):** `det_thresh=0.35` — tuned for recall, intentionally over-proposes
- **Stage 2 (Validator):** Geometric filter — discards anything that isn't structurally a face
- **Stage 2 costs < 1 ms** — adds precision without meaningful latency

---

## Why SCRFD-10GF

| Model | Easy AP | Medium AP | **Hard AP** | CPU Speed | Landmarks? |
|---|---|---|---|---|---|
| **SCRFD-10GF** | ~95.6% | ~94.2% | **~82.9%** | ~60–100ms | ✅ Always |
| SCRFD-34GF | ~96.1% | ~94.9% | ~85.3%+ | ~150ms | ✅ |
| RetinaFace R50 | ~94.9% | ~91.9% | ~64–91%* | ~200ms | ✅ |
| YOLOv8-Face | ~94.5% | ~92.2% | ~77–79% | ~30ms GPU | ⚠️ Inconsistent |
| MTCNN | ~88% | ~85% | ~70–78% | ~150ms | ✅ |
| YuNet | ~88% | ~86% | ~75% | ~2ms | ✅ |

Hard AP is the metric that matters for "find as many faces as possible" — SCRFD wins decisively. It also returns **5-point landmarks with every detection**, which is what makes Stage 2's geometric validation possible.

Use `buffalo_l` (InsightFace pack) which bundles SCRFD-10GF.

---

## Stage 1: Detection (SCRFD-10GF)

`det_thresh = 0.35` — lower than default 0.5. Stage 2 is the precision gate; Stage 1 just finds candidates.

### Tiling Strategy

For images > 1280px on any side, small faces get destroyed when downscaled to the model's 640×640 input. Tiling prevents this:

```
Original image (e.g., 3840×2160)
    ↓
Split into overlapping 640×640 tiles (20% overlap)
    ↓
Run SCRFD on each tile independently
    ↓
Remap bboxes/landmarks → original image coordinates
    ↓
Cross-tile NMS (IoU = 0.45) → remove duplicates
    ↓
Pass candidates to Stage 2
```

| Parameter | Value | Reason |
|---|---|---|
| `tile_size` | 640 | Matches SCRFD training resolution |
| `overlap` | 20% | Prevents faces cut at tile edges |
| `nms_iou_cross_tile` | 0.45 | Deduplicate across adjacent tiles |
| Skip tiling if | image ≤ 1280px both sides | No benefit; saves time |

---

## Stage 2: Validator (5 Layers, < 1 ms)

Every candidate must pass all five. One failure → rejected.

### L1 — Minimum Size
```python
if bbox_width < 20 or bbox_height < 20: reject()
```
Eliminates tiny noise: skin textures, background patterns.

### L2 — Aspect Ratio
```python
aspect = bbox_height / bbox_width
if not (0.5 <= aspect <= 2.2): reject()
```
Arms produce extremely elongated boxes (e.g., 300×60 = ratio 5.0) → instantly rejected.

### L3 — Landmark Geometry (4 checks)

SCRFD landmarks: `[left_eye, right_eye, nose_tip, mouth_left, mouth_right]`

- **3a** Eyes above nose: `eye_midpoint_y < nose_tip_y`
- **3b** Nose above mouth: `nose_tip_y < mouth_midpoint_y`
- **3c** Eye separation ratio: `0.15 ≤ eye_separation / bbox_width ≤ 0.85`
- **3d** Mouth center within lateral eye range: `mouth_center_x` between eyes ± margin

A random skin patch or arm gets "landmarks" placed arbitrarily — cannot satisfy all four simultaneously.

### L4 — Final Confidence Floor
```python
if confidence < 0.55: reject()
```
Stricter than Stage 1's 0.35. Borderline model guesses that pass geometry still need sufficient confidence.

### L5 — Final NMS Deduplication
```python
final = nms(validated_detections, iou_threshold=0.35)
```
Cleans up any remaining tile-boundary duplicates.

---

## Full Pipeline Diagram

```
Input Image (path / numpy / PIL)
         │
         ▼
    [Image Loader]
         │
         ▼
   [Adaptive Tiler] ── > 1280px → 640×640 overlapping tiles
         │                ≤ 1280px → single pass
         ▼
   [SCRFD-10GF]       det_thresh = 0.35
   (per tile/image)
         │
         ▼
  [Coord Remapper]    tile coords → original image coords
         │
         ▼
  [Cross-tile NMS]    iou = 0.45
         │
         ▼
━━━━━━ STAGE 2: VALIDATOR (< 1 ms total) ━━━━━━
   L1: Min size filter
   L2: Aspect ratio filter
   L3: Landmark geometry (4 checks)
   L4: Confidence floor ≥ 0.55
   L5: Final NMS deduplication
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         │
         ▼
  List[FaceDetection]
  { bbox, confidence, landmarks }
```

---

## File-by-File Code

### `face_detection/config.py`

```python
from dataclasses import dataclass

@dataclass
class DetectorConfig:
    # Model
    model_name: str = "buffalo_l"
    device: str = "cpu"                 # "cpu" | "cuda:0"

    # Stage 1
    det_thresh: float = 0.35
    nms_iou_cross_tile: float = 0.45

    # Tiling
    use_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.20
    tiling_threshold_px: int = 1280

    # Stage 2 validator
    min_face_px: int = 20
    aspect_ratio_min: float = 0.5
    aspect_ratio_max: float = 2.2
    eye_sep_ratio_min: float = 0.15
    eye_sep_ratio_max: float = 0.85
    final_confidence: float = 0.55
    final_nms_iou: float = 0.35

    # Output
    return_landmarks: bool = True
    return_confidence: bool = True
```

---

### `face_detection/loader.py`

```python
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def load_image(source) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Cannot load: {source}")
        return img
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        if source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
        return source.copy()
    elif isinstance(source, Image.Image):
        arr = np.array(source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Unsupported type: {type(source)}")
```

---

### `face_detection/tiling.py`

```python
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class Tile:
    image: np.ndarray
    x_offset: int
    y_offset: int

def generate_tiles(image: np.ndarray, tile_size: int, overlap: float) -> List[Tile]:
    H, W = image.shape[:2]
    step = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while True:
        x = 0
        while True:
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            tiles.append(Tile(image=image[y:y2, x:x2], x_offset=x, y_offset=y))
            if x2 == W:
                break
            x += step
        if y + tile_size >= H:
            break
        y += step
    return tiles

def remap_to_original(detections: list, tile: Tile) -> list:
    remapped = []
    for det in detections:
        b = det["bbox"]
        remapped.append({
            "bbox": [b[0]+tile.x_offset, b[1]+tile.y_offset,
                     b[2]+tile.x_offset, b[3]+tile.y_offset],
            "confidence": det["confidence"],
            "landmarks": (
                [[p[0]+tile.x_offset, p[1]+tile.y_offset] for p in det["landmarks"]]
                if det.get("landmarks") else None
            ),
        })
    return remapped
```

---

### `face_detection/postprocess.py`

```python
from typing import List

def iou(a: list, b: list) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def nms(detections: List[dict], iou_threshold: float) -> List[dict]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections
                      if iou(best["bbox"], d["bbox"]) < iou_threshold]
    return kept
```

---

### `face_detection/validator.py`

```python
from typing import List
from face_detection.config import DetectorConfig

def _validate(det: dict, cfg: DetectorConfig) -> bool:
    x1, y1, x2, y2 = det["bbox"]
    w, h = x2 - x1, y2 - y1
    conf = det["confidence"]
    lm   = det.get("landmarks")

    # L1: Minimum size
    if w < cfg.min_face_px or h < cfg.min_face_px:
        return False

    # L2: Aspect ratio
    if not (cfg.aspect_ratio_min <= h / (w + 1e-6) <= cfg.aspect_ratio_max):
        return False

    # L3: Landmark geometry
    if lm and len(lm) == 5:
        le, re, nt, ml, mr = lm
        eye_mid_y   = (le[1] + re[1]) / 2
        mouth_mid_y = (ml[1] + mr[1]) / 2
        eye_sep     = abs(re[0] - le[0])
        mouth_cx    = (ml[0] + mr[0]) / 2
        margin      = eye_sep * 0.30

        if not (eye_mid_y < nt[1]):                                      return False  # 3a
        if not (nt[1] < mouth_mid_y):                                    return False  # 3b
        if not (cfg.eye_sep_ratio_min <= eye_sep/(w+1e-6)
                                      <= cfg.eye_sep_ratio_max):          return False  # 3c
        if not (min(le[0],re[0]) - margin <= mouth_cx
                                           <= max(le[0],re[0]) + margin): return False  # 3d

    # L4: Final confidence
    if conf < cfg.final_confidence:
        return False

    return True

def apply_validator(detections: List[dict], cfg: DetectorConfig) -> List[dict]:
    return [d for d in detections if _validate(d, cfg)]
```

---

### `face_detection/models/scrfd.py`

```python
import numpy as np
from insightface.app import FaceAnalysis
from face_detection.config import DetectorConfig

class SCRFDDetector:
    def __init__(self, cfg: DetectorConfig):
        ctx_id = -1 if cfg.device == "cpu" else int(cfg.device.split(":")[-1])
        self.app = FaceAnalysis(
            name=cfg.model_name,
            allowed_modules=["detection"],  # skip recognizer — saves memory
        )
        self.app.prepare(ctx_id=ctx_id, det_thresh=cfg.det_thresh, det_size=(640, 640))

    def detect(self, image: np.ndarray) -> list:
        results = []
        for face in self.app.get(image):
            results.append({
                "bbox":       face.bbox.astype(int).tolist(),
                "confidence": float(face.det_score),
                "landmarks":  face.kps.tolist() if face.kps is not None else None,
            })
        return results
```

---

### `face_detection/detector.py`

```python
import numpy as np
from typing import Union, List
from pathlib import Path
from PIL import Image as PILImage

from face_detection.config import DetectorConfig
from face_detection.loader import load_image
from face_detection.models.scrfd import SCRFDDetector
from face_detection.tiling import generate_tiles, remap_to_original
from face_detection.postprocess import nms
from face_detection.validator import apply_validator


class FaceDetector:
    def __init__(self, config: DetectorConfig = None):
        self.cfg = config or DetectorConfig()
        self._model = SCRFDDetector(self.cfg)

    def detect(self, image: Union[str, Path, np.ndarray, PILImage.Image]) -> List[dict]:
        """
        Returns:
            List of { "bbox": [x1,y1,x2,y2], "confidence": float, "landmarks": [[x,y]x5] }
        """
        img = load_image(image)
        H, W = img.shape[:2]

        use_tiling = self.cfg.use_tiling and (
            W > self.cfg.tiling_threshold_px or H > self.cfg.tiling_threshold_px
        )

        if use_tiling:
            raw = []
            for tile in generate_tiles(img, self.cfg.tile_size, self.cfg.tile_overlap):
                raw.extend(remap_to_original(self._model.detect(tile.image), tile))
        else:
            raw = self._model.detect(img)

        raw       = nms(raw, self.cfg.nms_iou_cross_tile)
        validated = apply_validator(raw, self.cfg)
        final     = nms(validated, self.cfg.final_nms_iou)

        if not self.cfg.return_landmarks:
            for d in final:
                d.pop("landmarks", None)

        return final
```

---

### `face_detection/__init__.py`

```python
from face_detection.detector import FaceDetector
from face_detection.config import DetectorConfig
__all__ = ["FaceDetector", "DetectorConfig"]
```

---

### `face_detection/utils.py`

```python
import cv2
import numpy as np
from typing import List

LM_COLORS = [(255,0,0),(0,0,255),(0,255,255),(255,0,255),(255,165,0)]

def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    out = image.copy()
    for det in detections:
        x1,y1,x2,y2 = [int(v) for v in det["bbox"]]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"{det.get('confidence',0):.2f}",
                    (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        if det.get("landmarks"):
            for i,(lx,ly) in enumerate(det["landmarks"]):
                cv2.circle(out, (int(lx),int(ly)), 3, LM_COLORS[i%5], -1)
    return out
```

---

## Frontend — Minimalist Drag-and-Drop Tester

A single-page Flask app. Drop an image → receive it back annotated with bounding boxes + face count.

### `app/server.py`

```python
import io, base64, cv2, numpy as np
from flask import Flask, request, jsonify, send_from_directory
from face_detection import FaceDetector
from face_detection.utils import draw_detections

app      = Flask(__name__, static_folder="static")
detector = FaceDetector()   # model loads once at startup

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/detect", methods=["POST"])
def detect():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    nparr = np.frombuffer(file.read(), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    faces     = detector.detect(img)
    annotated = draw_detections(img, faces)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64    = base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "face_count": len(faces),
        "faces": [
            {"bbox": f["bbox"], "confidence": round(f["confidence"], 4)}
            for f in faces
        ],
        "annotated_image": f"data:image/jpeg;base64,{b64}",
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
```

---

### `app/static/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Face Detector</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <div class="container">
    <h1>🔍 Face Detector</h1>
    <p class="sub">Drop an image or click to upload</p>

    <div id="drop-zone" class="drop-zone">
      <span id="drop-text">📂 Drop image here</span>
      <input type="file" id="file-input" accept="image/*" hidden />
    </div>

    <div id="status" class="status hidden"></div>

    <div id="result-area" class="result-area hidden">
      <div class="result-header">
        <span id="face-count-label"></span>
        <button id="reset-btn">↩ Try another</button>
      </div>
      <div class="image-wrapper">
        <img id="result-img" alt="Annotated result" />
      </div>
      <div id="face-list"></div>
    </div>
  </div>
  <script src="main.js"></script>
</body>
</html>
```

---

### `app/static/style.css`

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: system-ui, sans-serif;
  background: #0f0f0f;
  color: #e8e8e8;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
}

.container { width: 100%; max-width: 820px; }

h1 { font-size: 1.8rem; margin-bottom: 6px; }
.sub { color: #888; margin-bottom: 24px; font-size: 0.95rem; }

.drop-zone {
  border: 2px dashed #444;
  border-radius: 12px;
  padding: 60px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
  font-size: 1.1rem;
  color: #aaa;
}
.drop-zone.drag-over {
  border-color: #4ade80;
  background: rgba(74,222,128,0.05);
  color: #4ade80;
}
.drop-zone:hover { border-color: #666; }

.status {
  margin-top: 20px;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 0.95rem;
  background: #1e1e1e;
  color: #aaa;
  text-align: center;
}
.status.error { background: #2a1010; color: #f87171; }

.result-area { margin-top: 28px; }

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 14px;
}

#face-count-label { font-size: 1.2rem; font-weight: 600; color: #4ade80; }

#reset-btn {
  background: #1e1e1e;
  border: 1px solid #333;
  color: #ccc;
  padding: 6px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background 0.15s;
}
#reset-btn:hover { background: #2a2a2a; }

.image-wrapper {
  border-radius: 10px;
  overflow: hidden;
  background: #1a1a1a;
  text-align: center;
}
#result-img { max-width: 100%; max-height: 70vh; display: block; margin: 0 auto; }

#face-list {
  margin-top: 18px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}
.face-card {
  background: #1a1a1a;
  border: 1px solid #2a2a2a;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 0.82rem;
  color: #aaa;
}
.face-card strong { color: #e8e8e8; }

.hidden { display: none !important; }
```

---

### `app/static/main.js`

```javascript
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const status     = document.getElementById("status");
const resultArea = document.getElementById("result-area");
const resultImg  = document.getElementById("result-img");
const faceCount  = document.getElementById("face-count-label");
const faceList   = document.getElementById("face-list");
const resetBtn   = document.getElementById("reset-btn");

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) uploadFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) uploadFile(file);
});

resetBtn.addEventListener("click", () => {
  resultArea.classList.add("hidden");
  status.classList.add("hidden");
  dropZone.classList.remove("hidden");
  fileInput.value = "";
});

async function uploadFile(file) {
  dropZone.classList.add("hidden");
  resultArea.classList.add("hidden");
  faceList.innerHTML = "";
  showStatus("⏳ Detecting faces…");

  const formData = new FormData();
  formData.append("image", file);

  try {
    const res  = await fetch("/detect", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showStatus(data.error || "Server error", true);
      return;
    }

    resultImg.src   = data.annotated_image;
    faceCount.textContent = `${data.face_count} face${data.face_count !== 1 ? "s" : ""} detected`;

    data.faces.forEach((f, i) => {
      const card = document.createElement("div");
      card.className = "face-card";
      card.innerHTML = `
        <strong>Face ${i+1}</strong><br/>
        Confidence: ${(f.confidence * 100).toFixed(1)}%<br/>
        Box: [${f.bbox.join(", ")}]
      `;
      faceList.appendChild(card);
    });

    status.classList.add("hidden");
    resultArea.classList.remove("hidden");

  } catch (err) {
    showStatus("Network error: " + err.message, true);
  }
}

function showStatus(msg, isError = false) {
  status.textContent = msg;
  status.className   = "status" + (isError ? " error" : "");
  status.classList.remove("hidden");
}
```

---

### `run.py`

```python
from app.server import app

if __name__ == "__main__":
    print("Face Detection Server → http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
```

---

## `requirements.txt`

```
# Detection
insightface>=0.7.3
onnxruntime>=1.16.0          # CPU
# onnxruntime-gpu>=1.16.0   # Uncomment if CUDA available

# Image
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.0.0

# Web server
flask>=3.0.0
```

---

## Install & Run

```bash
pip install -r requirements.txt
python run.py
# → Open http://localhost:5000
```

> [!NOTE]
> First run downloads `buffalo_l` model weights (~300 MB) to `~/.insightface/models/buffalo_l/` automatically. One-time only.

---

## Configuration Tuning Reference

| If you see... | Tweak | Direction |
|---|---|---|
| False positives (arms, skin patches) | `final_confidence` | ↑ raise (0.55 → 0.65) |
| False positives, tiny ones | `min_face_px` | ↑ raise (20 → 40) |
| Missing real faces (small/far) | `det_thresh` | ↓ lower (0.35 → 0.25) |
| Missing profile/tilted faces | `aspect_ratio_max` | ↑ raise (2.2 → 2.6) |
| Duplicate boxes on same face | `final_nms_iou` | ↓ lower (0.35 → 0.25) |
| Adjacent faces merged into one | `final_nms_iou` | ↑ raise (0.35 → 0.50) |
| Slow on large images | `tile_size` | ↑ raise (640 → 1024) |

---

## Verification — Test Image Checklist

| Image | Expected |
|---|---|
| Single close-up portrait | 1 face, high confidence |
| Group photo (5–15 people) | All faces, no limb false positives |
| Crowd photo (50+ people) | Tiling activates, maximum recall |
| Image with arms/hands prominent | Zero false positives on limbs |
| Small/distant faces | Tiling preserves them |
| Profile / side-angle face | Detected (aspect ratio accommodates profiles) |
