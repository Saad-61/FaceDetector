"""
server.py — Flask web server for the face detection tester.

Routes:
    GET  /          → serve index.html
    POST /detect    → detect faces, return annotated image + per-face crops
    GET  /health    → health check
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
    device="cpu",
    use_tiling=True,
    return_landmarks=True,
    return_confidence=True,
)
detector = FaceDetector(config=cfg)
print("[server] FaceDetector ready.")

_CROP_SIZE    = 160
_CROP_PADDING = 0.20


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Supported formats: JPG, PNG, WEBP, BMP."}), 400

    t0 = time.perf_counter()
    faces = detector.detect(img_bgr)
    t_detect = (time.perf_counter() - t0) * 1000

    annotated = draw_detections(img_bgr, faces)
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return jsonify({"error": "Failed to encode result image."}), 500
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    face_cards = []
    for face in faces:
        crop_bgr = _crop_face(img_bgr, face["bbox"])
        ok, cbuf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(cbuf).decode("utf-8") if ok else None

        card = {
            "bbox": face["bbox"],
            "crop": crop_b64,
        }
        if "confidence" in face:
            card["confidence"] = round(float(face["confidence"]), 4)
        if "landmarks" in face:
            card["landmarks"] = face["landmarks"]
        face_cards.append(card)

    return jsonify({
        "face_count": len(faces),
        "processing_ms": round(t_detect, 1),
        "annotated_image": annotated_b64,
        "faces": face_cards,
    })


def _crop_face(img: np.ndarray, bbox: list) -> np.ndarray:
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * _CROP_PADDING)
    py = int(bh * _CROP_PADDING)

    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(iw, x2 + px)
    cy2 = min(ih, y2 + py)

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        crop = np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
    else:
        crop = cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE), interpolation=cv2.INTER_LINEAR)

    return crop
