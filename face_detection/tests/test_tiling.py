"""
test_tiling.py — Unit tests for tile generation and coordinate remapping.

Run:
    python -m pytest face_detection/tests/test_tiling.py -v
"""

import numpy as np
import pytest
from face_detection.tiling import generate_tiles, remap_to_original, Tile


# ─── generate_tiles ────────────────────────────────────────────────────────────

def _blank(h, w):
    return np.zeros((h, w, 3), dtype=np.uint8)

def test_single_tile_small_image():
    # Image smaller than tile_size → should produce exactly 1 tile
    img = _blank(400, 400)
    tiles = generate_tiles(img, tile_size=640, overlap=0.20)
    assert len(tiles) == 1
    assert tiles[0].x_offset == 0
    assert tiles[0].y_offset == 0

def test_tiles_cover_entire_image():
    # Every pixel must be covered by at least one tile
    H, W = 1000, 1000
    tile_size = 640
    overlap = 0.20
    img = _blank(H, W)
    tiles = generate_tiles(img, tile_size=tile_size, overlap=overlap)

    covered = np.zeros((H, W), dtype=bool)
    for t in tiles:
        th, tw = t.image.shape[:2]
        covered[t.y_offset:t.y_offset + th, t.x_offset:t.x_offset + tw] = True

    assert covered.all(), "Some pixels are not covered by any tile"

def test_tile_images_have_correct_shape():
    img = _blank(800, 1200)
    tiles = generate_tiles(img, tile_size=640, overlap=0.20)
    for t in tiles:
        assert t.image.ndim == 3
        assert t.image.shape[2] == 3


# ─── remap_to_original ─────────────────────────────────────────────────────────

def _make_tile(x_off, y_off):
    img = _blank(640, 640)
    return Tile(image=img, x_offset=x_off, y_offset=y_off)

def test_remap_bbox_no_offset():
    tile = _make_tile(0, 0)
    dets = [{"bbox": [10, 20, 80, 90], "confidence": 0.9, "landmarks": None}]
    result = remap_to_original(dets, tile)
    assert result[0]["bbox"] == [10, 20, 80, 90]

def test_remap_bbox_with_offset():
    tile = _make_tile(x_off=200, y_off=150)
    dets = [{"bbox": [10, 20, 80, 90], "confidence": 0.9, "landmarks": None}]
    result = remap_to_original(dets, tile)
    assert result[0]["bbox"] == [210, 170, 280, 240]

def test_remap_landmarks_with_offset():
    tile = _make_tile(x_off=100, y_off=50)
    lm = [[30.0, 40.0], [60.0, 40.0], [45.0, 60.0], [32.0, 75.0], [58.0, 75.0]]
    dets = [{"bbox": [20, 20, 100, 100], "confidence": 0.85, "landmarks": lm}]
    result = remap_to_original(dets, tile)

    expected_lm = [[l[0] + 100, l[1] + 50] for l in lm]
    assert result[0]["landmarks"] == expected_lm

def test_remap_preserves_confidence():
    tile = _make_tile(50, 50)
    dets = [{"bbox": [0, 0, 60, 60], "confidence": 0.77, "landmarks": None}]
    result = remap_to_original(dets, tile)
    assert result[0]["confidence"] == pytest.approx(0.77)

def test_remap_does_not_mutate_original():
    tile = _make_tile(100, 100)
    original_bbox = [10, 10, 50, 50]
    dets = [{"bbox": original_bbox[:], "confidence": 0.9, "landmarks": None}]
    remap_to_original(dets, tile)
    # original det bbox should not be modified
    assert dets[0]["bbox"] == [10, 10, 50, 50]
