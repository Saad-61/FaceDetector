"""
utils.py — Visualization helpers for face detection results.

These utilities are optional — not required for core detection.
Import and use them when you want to draw results on images for
debugging, testing, or the web frontend.
"""

import cv2
import numpy as np
from typing import List

# Colors for each of the 5 SCRFD landmarks (BGR format for OpenCV)
_LANDMARK_COLORS = [
    (255,  0,   0),    # 0: left eye       → blue
    (  0,  0, 255),    # 1: right eye      → red
    (  0, 255, 255),   # 2: nose tip       → yellow
    (255,  0, 255),    # 3: mouth left     → magenta
    (255, 165,  0),    # 4: mouth right    → orange
]


def draw_detections(image: np.ndarray, detections: List[dict]) -> np.ndarray:
    """
    Draw bounding boxes, confidence scores, and landmarks on a copy of image.

    Args:
        image:      BGR numpy array (H, W, 3). Not modified in-place.
        detections: List of face dicts from FaceDetector.detect().

    Returns:
        Annotated copy of the image (BGR numpy array).
    """
    out = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        conf = det.get("confidence", 0.0)

        # ── Bounding box ──────────────────────────────────────────────────
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ── Confidence label ──────────────────────────────────────────────
        label = f"{conf:.2f}"
        label_y = y1 - 8 if y1 > 20 else y2 + 18
        cv2.putText(
            out, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA
        )

        # ── Landmarks ─────────────────────────────────────────────────────
        lm = det.get("landmarks")
        if lm:
            for i, (lx, ly) in enumerate(lm):
                color = _LANDMARK_COLORS[i % len(_LANDMARK_COLORS)]
                cv2.circle(out, (int(lx), int(ly)), 3, color, -1, cv2.LINE_AA)

    return out


def image_to_jpeg_bytes(image: np.ndarray, quality: int = 90) -> bytes:
    """
    Encode a BGR numpy array to JPEG bytes.

    Args:
        image:   BGR numpy array.
        quality: JPEG quality 1–100. Default 90.

    Returns:
        Raw JPEG bytes.
    """
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
