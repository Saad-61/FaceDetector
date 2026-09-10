"""
test_postprocess.py — Unit tests for IoU computation and NMS.

Run:
    python -m pytest face_detection/tests/test_postprocess.py -v
"""

import pytest
from face_detection.postprocess import iou, nms


# ─── IoU ───────────────────────────────────────────────────────────────────────

def test_iou_identical_boxes():
    box = [0, 0, 100, 100]
    assert iou(box, box) == pytest.approx(1.0)

def test_iou_no_overlap():
    a = [0,   0,  50,  50]
    b = [60, 60, 110, 110]
    assert iou(a, b) == pytest.approx(0.0)

def test_iou_partial_overlap():
    a = [0, 0, 100, 100]   # area 10000
    b = [50, 0, 150, 100]  # area 10000
    # intersection: [50,0,100,100] = 5000
    # union: 10000 + 10000 - 5000 = 15000
    assert iou(a, b) == pytest.approx(5000 / 15000)

def test_iou_contained_box():
    outer = [0, 0, 100, 100]
    inner = [25, 25, 75, 75]   # area 2500
    # intersection = 2500, union = 10000
    assert iou(outer, inner) == pytest.approx(2500 / 10000)


# ─── NMS ───────────────────────────────────────────────────────────────────────

def _det(bbox, conf):
    return {"bbox": bbox, "confidence": conf, "landmarks": None}

def test_nms_empty():
    assert nms([], 0.5) == []

def test_nms_single():
    d = _det([0, 0, 100, 100], 0.9)
    result = nms([d], 0.5)
    assert len(result) == 1

def test_nms_removes_duplicate():
    # Two nearly identical boxes — only the higher-confidence one should survive
    d1 = _det([0, 0, 100, 100], 0.95)
    d2 = _det([2, 2, 102, 102], 0.60)  # IoU ≈ 0.96 with d1 → suppressed
    result = nms([d1, d2], iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]["confidence"] == pytest.approx(0.95)

def test_nms_keeps_non_overlapping():
    d1 = _det([0,   0,  100, 100], 0.90)
    d2 = _det([200, 200, 300, 300], 0.85)  # no overlap
    result = nms([d1, d2], iou_threshold=0.5)
    assert len(result) == 2

def test_nms_order_by_confidence():
    # Lower confidence box comes first in input — NMS should still keep the higher one
    d_low  = _det([0, 0, 100, 100], 0.50)
    d_high = _det([5, 5, 105, 105], 0.90)
    result = nms([d_low, d_high], iou_threshold=0.5)
    assert len(result) == 1
    assert result[0]["confidence"] == pytest.approx(0.90)
