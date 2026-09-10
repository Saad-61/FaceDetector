"""
test_validator.py — Unit tests for the 5-layer precision validator.

Tests each layer in isolation using synthetic detection dicts.
No model or GPU required — pure logic tests.

Run:
    python -m pytest face_detection/tests/test_validator.py -v
"""

import pytest
from face_detection.config import DetectorConfig
from face_detection.validator import _validate_single

# Default config for tests
CFG = DetectorConfig()


def make_det(
    bbox=(100, 100, 200, 220),
    confidence=0.90,
    landmarks=None,
):
    """Build a synthetic detection dict. Default values represent a valid face."""
    if landmarks is None:
        # Geometrically valid 5-point landmarks for a 100×120 face at (100,100)
        landmarks = [
            [130.0, 140.0],   # left eye
            [170.0, 140.0],   # right eye
            [150.0, 165.0],   # nose tip
            [132.0, 185.0],   # mouth left
            [168.0, 185.0],   # mouth right
        ]
    return {"bbox": list(bbox), "confidence": confidence, "landmarks": landmarks}


# ─── L1: Minimum Size ──────────────────────────────────────────────────────────

def test_L1_passes_normal_size():
    assert _validate_single(make_det(bbox=(0, 0, 60, 80)), CFG) is True

def test_L1_fails_too_narrow():
    det = make_det(bbox=(0, 0, 10, 80))   # width = 10 < min_face_px=20
    assert _validate_single(det, CFG) is False

def test_L1_fails_too_short():
    det = make_det(bbox=(0, 0, 80, 15))   # height = 15 < min_face_px=20
    assert _validate_single(det, CFG) is False


# ─── L2: Aspect Ratio ──────────────────────────────────────────────────────────

def test_L2_passes_normal_aspect():
    det = make_det(bbox=(0, 0, 100, 130))  # h/w = 1.3 — normal face
    assert _validate_single(det, CFG) is True

def test_L2_fails_arm_shaped():
    # Very tall narrow box like an arm (h/w = 5.0)
    det = make_det(bbox=(0, 0, 40, 200))
    assert _validate_single(det, CFG) is False

def test_L2_fails_too_wide():
    # Very wide flat box (h/w = 0.3)
    det = make_det(bbox=(0, 0, 200, 60))
    assert _validate_single(det, CFG) is False


# ─── L3: Landmark Geometry ─────────────────────────────────────────────────────

def test_L3_passes_valid_landmarks():
    assert _validate_single(make_det(), CFG) is True

def test_L3_fails_eyes_below_nose():
    lm = [
        [130.0, 180.0],   # left eye — BELOW nose
        [170.0, 180.0],   # right eye — BELOW nose
        [150.0, 140.0],   # nose tip — ABOVE eyes (inverted)
        [132.0, 200.0],
        [168.0, 200.0],
    ]
    assert _validate_single(make_det(landmarks=lm), CFG) is False

def test_L3_fails_nose_below_mouth():
    lm = [
        [130.0, 130.0],
        [170.0, 130.0],
        [150.0, 200.0],   # nose tip — BELOW mouth (inverted)
        [132.0, 160.0],
        [168.0, 160.0],
    ]
    assert _validate_single(make_det(landmarks=lm), CFG) is False

def test_L3_fails_eyes_too_close():
    # Eye separation = 4px on a 100px wide box → ratio 0.04 < 0.15
    lm = [
        [148.0, 140.0],
        [152.0, 140.0],   # eyes only 4px apart
        [150.0, 165.0],
        [132.0, 185.0],
        [168.0, 185.0],
    ]
    assert _validate_single(make_det(landmarks=lm), CFG) is False

def test_L3_fails_mouth_outside_eyes():
    lm = [
        [130.0, 140.0],
        [170.0, 140.0],
        [150.0, 165.0],
        [300.0, 185.0],   # mouth far outside eye range
        [320.0, 185.0],
    ]
    assert _validate_single(make_det(landmarks=lm), CFG) is False


# ─── L4: Confidence Floor ──────────────────────────────────────────────────────

def test_L4_passes_above_threshold():
    assert _validate_single(make_det(confidence=0.80), CFG) is True

def test_L4_fails_below_threshold():
    assert _validate_single(make_det(confidence=0.40), CFG) is False

def test_L4_passes_exactly_at_threshold():
    # final_confidence default = 0.55; exactly at boundary → passes (>=)
    assert _validate_single(make_det(confidence=0.55), CFG) is True


# ─── Combined: valid detection end-to-end ─────────────────────────────────────

def test_full_valid_detection_passes():
    det = make_det(
        bbox=(50, 50, 170, 200),
        confidence=0.95,
    )
    assert _validate_single(det, CFG) is True
