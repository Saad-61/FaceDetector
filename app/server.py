"""
server.py — Flask web server for the face detection tester.

Routes:
    GET  /          → serve index.html
    POST /detect    → detect faces, return annotated image + per-face crops
    GET  /health    → health check

Response JSON:
{
    "face_count":      int,
    "processing_ms":   float,
    "annotated_image": "data:image/jpeg;base64,...",
    "faces": [
        {
            "bbox":       [x1, y1, x2, y2],
            "confidence": float,
            "crop":       "data:image/jpeg;base64,..."  ← cropped face thumbnail
        },
        ...
    ]
}
"""

import base64
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from face_detection import FaceDetector, DetectorConfig
from face_detection.utils import draw_detections

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")

cfg = DetectorConfig(
    device="cpu",            # switch to "cuda:0" if GPU available
    use_tiling=True,
    max_image_size=2560,     # caps large images → drastically reduces tile count
    return_landmarks=True,
    return_confidence=True,
)
detector = FaceDetector(config=cfg)
print("[server] FaceDetector ready.")

_CROP_SIZE    = 220    # thumbnail size in pixels (high quality square preview)
_CROP_PADDING = 0.20   # fractional padding around the bbox


# ── Routes ────────────────────────────────────────────────────────────────────

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning"
    return response


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/favicon.ico")
def favicon():
    resp = send_from_directory("static", "favicon.ico", mimetype="image/x-icon")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/favicon.svg")
def favicon_svg():
    resp = send_from_directory("static", "favicon.svg", mimetype="image/svg+xml")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/favicon.png")
def favicon_png():
    resp = send_from_directory("static", "favicon.png", mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    if request.method == "OPTIONS":
        return "", 204

    file = request.files.get("image")
    if file is None or not file.filename:
        print("[server] /detect 400: No image provided in 'image' field")
        return jsonify({"error": "No image provided. Field name must be 'image'."}), 400

    raw_bytes = file.read()
    if not raw_bytes:
        print(f"[server] /detect 400: Received 0 bytes for '{file.filename}' (upload incomplete or aborted)")
        return jsonify({"error": "Received empty file or upload was interrupted."}), 400

    nparr = np.frombuffer(raw_bytes, dtype=np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[server] /detect 400: Failed to decode {len(raw_bytes)} bytes as image")
        return jsonify({"error": "Could not decode image. Supported formats: JPG, PNG, WEBP, BMP."}), 400

    H, W = img.shape[:2]
    print(f"[server] /detect: incoming '{file.filename}', {len(raw_bytes)/1024:.1f} KB, resolution {W}x{H}")

    # Detect faces
    t0    = time.perf_counter()
    faces = detector.detect(img)
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    # Annotated image (bboxes drawn on original image)
    annotated = draw_detections(img, faces)

    # Web preview optimization:
    # Scale annotated preview down to max 1600px if larger.
    # This prevents transmitting 10-15 MB base64 payloads over ngrok tunnels.
    max_preview_dim = 1600
    if max(H, W) > max_preview_dim:
        scale = max_preview_dim / max(H, W)
        pw, ph = int(W * scale), int(H * scale)
        preview = cv2.resize(annotated, (pw, ph), interpolation=cv2.INTER_AREA)
    else:
        preview = annotated

    _, buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
    annotated_b64 = base64.b64encode(buf).decode("utf-8")

    # Per-face crops from original resolution image
    face_data = []
    for face in faces:
        crop_b64 = _crop_face(img, face["bbox"], H, W)
        face_data.append({
            "bbox":       face["bbox"],
            "confidence": round(face["confidence"], 4),
            "crop":       crop_b64,
        })

    print(f"[server] /detect: found {len(faces)} faces in {ms} ms. Preview payload: {len(annotated_b64)/1024:.1f} KB")

    return jsonify({
        "face_count":      len(faces),
        "processing_ms":   ms,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "faces":           face_data,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crop_face(img: np.ndarray, bbox: list, H: int, W: int) -> str:
    """
    Crop a face from img with natural proportions (no stretching/squishing),
    using high-quality Lanczos4 interpolation + subtle detail enhancement for small faces,
    and area interpolation for high-res faces.
    """
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # Human faces are taller than wide. Use the larger dimension + 28% padding
    # to construct a square region centered on the face.
    # This completely eliminates distortion/squashing of rectangular human faces into a square.
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(bw, bh) * 1.28

    cx1 = int(round(center_x - side / 2.0))
    cy1 = int(round(center_y - side / 2.0))
    cx2 = int(round(cx1 + side))
    cy2 = int(round(cy1 + side))

    # Pad if outside image boundaries with reflection
    pad_top = max(0, -cy1)
    pad_bottom = max(0, cy2 - H)
    pad_left = max(0, -cx1)
    pad_right = max(0, cx2 - W)

    crop = img[max(0, cy1):min(H, cy2), max(0, cx1):min(W, cx2)]
    if crop.size == 0:
        crop = np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
    else:
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)

        ch, cw = crop.shape[:2]
        if cw < _CROP_SIZE:
            # Small face: use LANCZOS4 high-order interpolation + subtle detail enhancement
            scaled = cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE), interpolation=cv2.INTER_LANCZOS4)
            # Unsharp mask to clarify facial features (eyes, nose, mouth)
            blur = cv2.GaussianBlur(scaled, (0, 0), 1.0)
            crop = cv2.addWeighted(scaled, 1.25, blur, -0.25, 0)
        else:
            # High-res face: use INTER_AREA to prevent aliasing
            crop = cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE), interpolation=cv2.INTER_AREA)

    # Encode with high quality (94 quality: crisp DCT, clean edges)
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
