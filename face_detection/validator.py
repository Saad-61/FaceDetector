"""
validator.py — Stage 2: Geometric & quality validator for face candidates.

The core design principle:
  SCRFD (Stage 1) is tuned for high recall, which means it over-proposes.
  Validator (Stage 2) is the precision gate.
  It discards anything that physically cannot be a human face.
  Total runtime for all 5 layers: < 1 ms per candidate.
"""

from typing import List, Dict, Any
import numpy as np
from face_detection.config import DetectorConfig
from face_detection.postprocess import nms


def validate_face(face: Dict[str, Any], cfg: DetectorConfig) -> bool:
    """
    Apply structural validity checks to a single candidate detection.
    Returns True if the candidate passes all checks, False otherwise.
    """
    x1, y1, x2, y2 = face["bbox"]
    w = x2 - x1
    h = y2 - y1

    # ── Layer 1: Minimum size ─────────────────────────────────────────────
    if w < cfg.min_face_px or h < cfg.min_face_px:
        return False

    # ── Layer 2: Aspect ratio ─────────────────────────────────────────────
    aspect_ratio = h / (w + 1e-6)
    if not (cfg.aspect_ratio_min <= aspect_ratio <= cfg.aspect_ratio_max):
        return False

    # ── Layer 3: 5-point landmark structural geometry ─────────────────────
    landmarks = face.get("landmarks")
    if landmarks is not None and len(landmarks) == 5:
        left_eye    = landmarks[0]
        right_eye   = landmarks[1]
        nose_tip    = landmarks[2]
        mouth_left  = landmarks[3]
        mouth_right = landmarks[4]

        # 3a: Eyes must be above the nose
        if not (nose_tip[1] > min(left_eye[1], right_eye[1])):
            return False

        # 3b: Mouth must be below the nose
        mouth_cy = (mouth_left[1] + mouth_right[1]) / 2.0
        if not (mouth_cy > nose_tip[1]):
            return False

        # 3c: Eye separation must be a plausible fraction of bbox width
        eye_sep   = abs(right_eye[0] - left_eye[0])
        sep_ratio = eye_sep / (w + 1e-6)
        if not (cfg.eye_sep_ratio_min <= sep_ratio <= cfg.eye_sep_ratio_max):
            return False

        # 3d: Mouth center must sit between the eyes laterally
        margin      = eye_sep * 0.50
        left_bound  = min(left_eye[0], right_eye[0]) - margin
        right_bound = max(left_eye[0], right_eye[0]) + margin
        mouth_cx    = (mouth_left[0] + mouth_right[0]) / 2.0
        if not (left_bound <= mouth_cx <= right_bound):
            return False

    # ── Layer 4: Final confidence floor ───────────────────────────────────
    if face.get("confidence", 1.0) < cfg.final_confidence:
        return False

    return True


def validate_and_deduplicate(
    faces: List[Dict[str, Any]],
    cfg: DetectorConfig,
) -> List[Dict[str, Any]]:
    """
    Run all candidate detections through the Stage 2 pipeline:
      1. Filter through Layers 1–4 (validate_face)
      2. Layer 5: Final NMS deduplication
    """
    valid = [f for f in faces if validate_face(f, cfg)]
    final = nms(valid, cfg.nms_iou_final)
    return final
