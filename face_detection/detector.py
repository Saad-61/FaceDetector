"""
detector.py — FaceDetector: the main orchestrator and public API.

Usage:
    from face_detection import FaceDetector, DetectorConfig

    detector = FaceDetector()
    faces    = detector.detect("photo.jpg")   # returns list of face dicts

Pipeline (in order):
    1. load_image()          — normalize input to BGR numpy array
    2. _maybe_resize()       — cap very large images to max_image_size (SPEED FIX)
    3. generate_tiles()      — split into overlapping tiles if image still large
    4. SCRFDDetector.detect()— run SCRFD-10GF on each tile/image
    5. remap_to_original()   — tile-local coords → image coords
    6. nms()                 — cross-tile deduplication (iou=0.45)
    7. apply_validator()     — 5-layer precision filter (< 1 ms)
    8. nms()                 — final deduplication (iou=0.35)
    9. scale back            — if resized, map bboxes back to original coords
"""

import cv2
import numpy as np
from typing import Union, List, Tuple
from pathlib import Path

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

from face_detection.config import DetectorConfig
from face_detection.loader import load_image
from face_detection.models.scrfd import SCRFDDetector
from face_detection.tiling import generate_tiles, remap_to_original
from face_detection.postprocess import nms
from face_detection.validator import apply_validator


class FaceDetector:
    """
    High-recall, high-precision face detector.

    Re-use one instance for multiple detect() calls — the model loads once
    at __init__ and is kept in memory.
    """

    def __init__(self, config: DetectorConfig = None):
        """
        Load the SCRFD model.

        Args:
            config: DetectorConfig. Uses defaults if None.

        Note:
            First run downloads buffalo_l weights (~300 MB) to ~/.insightface/models/.
        """
        self.cfg = config or DetectorConfig()
        self._model = SCRFDDetector(self.cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        image: Union[str, Path, np.ndarray, "PILImage.Image"],
    ) -> List[dict]:
        """
        Detect all human faces in an image.

        Args:
            image: file path (str/Path), numpy array (BGR/RGB), or PIL Image.

        Returns:
            List of dicts sorted by confidence (descending):
            {
                "bbox":       [x1, y1, x2, y2],  # pixel coords in ORIGINAL image
                "confidence": float,              # 0.0–1.0
                "landmarks":  [[x,y], ...]        # 5 keypoints, or None
            }
        """
        img = load_image(image)

        # Cap very large images before detection — major speed win.
        # bboxes are scaled back to original coords at the end.
        img_small, scale = self._maybe_resize(img)

        H, W = img_small.shape[:2]
        should_tile = (
            self.cfg.use_tiling
            and (W > self.cfg.tiling_threshold_px or H > self.cfg.tiling_threshold_px)
        )

        # Stage 1: Detection
        if should_tile:
            raw = self._detect_with_tiling(img_small)
        else:
            raw = self._model.detect(img_small)

        raw = nms(raw, self.cfg.nms_iou_cross_tile)

        # Stage 2: Validator
        validated = apply_validator(raw, self.cfg)
        final     = nms(validated, self.cfg.final_nms_iou)

        # Scale bboxes/landmarks back to original image coordinates
        if scale != 1.0:
            final = self._scale_back(final, 1.0 / scale)

        if not self.cfg.return_landmarks:
            for d in final:
                d.pop("landmarks", None)

        return final

    # ── Private helpers ───────────────────────────────────────────────────────

    def _maybe_resize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Proportionally downscale img if its longest side > cfg.max_image_size.

        Returns (resized_img, scale). scale == 1.0 means no resize happened.

        Speed impact example:
            6000×4000 px, 640px tiles, 20% overlap → ~96 tiles → ~28s CPU
            Capped at 2560px              → ~16 tiles → ~5s  CPU
        """
        max_side = self.cfg.max_image_size
        if max_side <= 0:
            return img, 1.0

        H, W = img.shape[:2]
        longest = max(H, W)
        if longest <= max_side:
            return img, 1.0

        scale   = max_side / longest
        new_w   = int(W * scale)
        new_h   = int(H * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    @staticmethod
    def _scale_back(detections: List[dict], inv_scale: float) -> List[dict]:
        """Multiply all bbox and landmark coordinates by inv_scale."""
        for d in detections:
            b = d["bbox"]
            d["bbox"] = [
                int(b[0] * inv_scale),
                int(b[1] * inv_scale),
                int(b[2] * inv_scale),
                int(b[3] * inv_scale),
            ]
            if d.get("landmarks"):
                d["landmarks"] = [
                    [pt[0] * inv_scale, pt[1] * inv_scale]
                    for pt in d["landmarks"]
                ]
        return detections

    def _detect_with_tiling(self, img: np.ndarray) -> List[dict]:
        """Run SCRFD on each tile and remap results to img's coordinate space."""
        tiles = generate_tiles(img, self.cfg.tile_size, self.cfg.tile_overlap)
        all_detections: List[dict] = []
        for tile in tiles:
            tile_dets = self._model.detect(tile.image)
            all_detections.extend(remap_to_original(tile_dets, tile))
        return all_detections
