"""
validator.py — 5-Layer precision filter (Stage 2 of the detection pipeline).

Purpose:
  SCRFD (Stage 1) is tuned for high recall, which means it over-proposes.
  This validator discards detections that are geometrically implausible as
  human faces — arms, skin patches, hands, background textures, etc.

All 5 layers run in < 1 ms regardless of how many candidates there are,
because they operate on a short list of Python dicts (no neural network,
no GPU, no memory allocation).

Layers:
  L1 — Minimum Size:        bbox too small → noise / artifact
  L2 — Aspect Ratio:        bbox too wide or too narrow → arms, limbs
  L3 — Landmark Geometry:   4 checks that only real faces can satisfy
  L4 — Confidence Floor:    post-geometry confidence threshold (stricter)
  L5 — Final NMS:           called externally in detector.py (not here)

Landmark layout from SCRFD:
  lm[0] = left_eye
  lm[1] = right_eye
  lm[2] = nose_tip
  lm[3] = mouth_left_corner
  lm[4] = mouth_right_corner
"""

from typing import List
from face_detection.config import DetectorConfig


def _validate_single(det: dict, cfg: DetectorConfig) -> bool:
    """
    Run all validator layers on a single detection dict.
    Returns True if the detection passes all layers (keep it).
    Returns False on the first layer that fails (discard it).
    """
    x1, y1, x2, y2 = det["bbox"]
    w = x2 - x1
    h = y2 - y1
    conf = det["confidence"]
    lm = det.get("landmarks")

    # ── L1: Minimum Size ──────────────────────────────────────────────────
    # Both width and height must exceed the minimum.
    # Small detections are almost always noise, not distant faces
    # (SCRFD tiling already handles small-but-real distant faces).
    if w < cfg.min_face_px or h < cfg.min_face_px:
        return False

    # ── L2: Aspect Ratio ──────────────────────────────────────────────────
    # Real face bounding boxes have aspect ratio (h/w) between ~0.5 and ~2.2.
    # Arms / forearms → very tall and narrow (ratio 3–6) → rejected.
    # Wide flat patches → ratio < 0.5 → rejected.
    aspect = h / (w + 1e-6)
    if not (cfg.aspect_ratio_min <= aspect <= cfg.aspect_ratio_max):
        return False

    # ── L3: Landmark Geometry ─────────────────────────────────────────────
    # Only runs if landmarks are present (SCRFD always provides them).
    # Four geometric invariants that every real face satisfies:
    if lm is not None and len(lm) == 5:
        left_eye    = lm[0]
        right_eye   = lm[1]
        nose_tip    = lm[2]
        mouth_left  = lm[3]
        mouth_right = lm[4]

        eye_mid_y   = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (mouth_left[1] + mouth_right[1]) / 2.0
        eye_sep     = abs(right_eye[0] - left_eye[0])
        mouth_cx    = (mouth_left[0] + mouth_right[0]) / 2.0
        sep_ratio   = eye_sep / (w + 1e-6)

        # Vertical tolerance to accommodate tilted heads, screaming/open mouths,
        # or hands partially occluding the mouth/chin:
        v_tol = max(2.0, eye_sep * 0.15)

        # 3a: Eyes must be above the nose (y increases downward)
        if not (eye_mid_y < nose_tip[1] + v_tol):
            return False

        # 3b: Nose must be above the mouth
        if not (nose_tip[1] < mouth_mid_y + v_tol):
            return False

        mouth_sep   = abs(mouth_right[0] - mouth_left[0])
        mouth_ratio = mouth_sep / (w + 1e-6)

        # Check if the face is viewed in profile (yaw angle > 45-60°).
        # In profile perspective, both eyes and mouth features are foreshortened
        # horizontally along the line of sight.
        is_profile = (sep_ratio < 0.20) and (mouth_ratio < 0.25)

        if is_profile:
            # Profile view: mouth and nose must lie within the bounding box horizontally
            margin_x = w * 0.15
            if not (x1 - margin_x <= mouth_cx <= x2 + margin_x):
                return False
        else:
            # Frontal / semi-frontal view:
            # 3c: Eye separation must be a plausible fraction of bbox width
            if not (cfg.eye_sep_ratio_min <= sep_ratio <= cfg.eye_sep_ratio_max):
                return False

            # 3d: Mouth center must sit between the eyes laterally (with 50% margin for 3/4 poses)
            margin      = eye_sep * 0.50
            left_bound  = min(left_eye[0], right_eye[0]) - margin
            right_bound = max(left_eye[0], right_eye[0]) + margin
            if not (left_bound <= mouth_cx <= right_bound):
                return False

    # ── L4: Final Confidence Floor ────────────────────────────────────────
    # Stricter than Stage 1's det_thresh (0.35).
    # Detections that pass geometry but have borderline model confidence
    # are still discarded here.
    if conf < cfg.final_confidence:
        return False

    # ── Passed all layers ─────────────────────────────────────────────────
    return True


def apply_validator(detections: List[dict], cfg: DetectorConfig) -> List[dict]:
    """
    Apply the full 5-layer validator to a list of detections.

    Args:
        detections: List of raw detection dicts from the model / tiler.
        cfg:        DetectorConfig instance with all threshold values.

    Returns:
        Filtered list containing only detections that passed all layers.
    """
    return [d for d in detections if _validate_single(d, cfg)]
