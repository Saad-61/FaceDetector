"""
tiling.py — Sliding window tile generation and coordinate remapping.

Strategy:
  - Split large images into overlapping tiles (default 640×640, 20% overlap).
  - Run the detector on each tile independently at its native resolution.
  - Remap tile-local bounding boxes and landmarks back to original image space.
  - Cross-tile NMS (in postprocess.py) then removes duplicates.

This prevents small/distant faces from being destroyed when a large image
is downscaled to fit the model's fixed input resolution.
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class Tile:
    """A single image tile with its offset in the original image."""
    image: np.ndarray   # cropped tile, BGR uint8
    x_offset: int       # tile's top-left x-coordinate in the original image
    y_offset: int       # tile's top-left y-coordinate in the original image


def generate_tiles(image: np.ndarray, tile_size: int, overlap: float) -> List[Tile]:
    """
    Divide image into overlapping square tiles using a sliding window.

    Args:
        image:     BGR numpy array (H, W, 3).
        tile_size: Size of each tile in pixels (both width and height).
        overlap:   Fraction of tile_size used as overlap (e.g., 0.20 = 20%).

    Returns:
        List of Tile objects covering the entire image.
        Tiles at right/bottom edges may be smaller than tile_size if
        the image doesn't divide evenly.
    """
    H, W = image.shape[:2]
    step = int(tile_size * (1.0 - overlap))  # pixels between tile starts

    if step <= 0:
        raise ValueError(
            f"overlap={overlap} is too large for tile_size={tile_size}. "
            "Use overlap < 1.0."
        )

    tiles: List[Tile] = []

    y = 0
    while True:
        x = 0
        while True:
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)

            crop = image[y:y2, x:x2]
            tiles.append(Tile(image=crop, x_offset=x, y_offset=y))

            if x2 >= W:
                break
            x += step

        if y2 >= H:
            break
        y += step

    return tiles


def remap_to_original(detections: list, tile: Tile) -> list:
    """
    Shift bounding boxes and landmarks from tile-local coordinates
    to the original image's coordinate space.

    Args:
        detections: List of detection dicts from the model (tile-local coords).
        tile:       The Tile object with x_offset, y_offset in original image.

    Returns:
        New list of detection dicts with coordinates in original image space.
        Original dicts are not modified.
    """
    remapped = []
    for det in detections:
        b = det["bbox"]  # [x1, y1, x2, y2] in tile space

        # Shift bbox
        new_bbox = [
            b[0] + tile.x_offset,
            b[1] + tile.y_offset,
            b[2] + tile.x_offset,
            b[3] + tile.y_offset,
        ]

        # Shift landmarks (if present)
        new_landmarks = None
        if det.get("landmarks") is not None:
            new_landmarks = [
                [pt[0] + tile.x_offset, pt[1] + tile.y_offset]
                for pt in det["landmarks"]
            ]

        remapped.append({
            "bbox":       new_bbox,
            "confidence": det["confidence"],
            "landmarks":  new_landmarks,
        })

    return remapped
