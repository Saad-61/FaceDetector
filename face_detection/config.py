from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    """
    Central configuration for FaceDetector.
    All tunable values live here — nothing is hardcoded elsewhere.
    Edit thresholds here to adjust the precision/recall trade-off.
    """

    # ── Model ─────────────────────────────────────────────────────────────
    model_name: str = "buffalo_l"
    # InsightFace model pack. "buffalo_l" bundles SCRFD-10GF (detection)
    # + ArcFace recognition. We load detection-only via allowed_modules.

    device: str = "cpu"
    # "cpu"     → runs on CPU via onnxruntime
    # "cuda:0"  → runs on first GPU via onnxruntime-gpu
    # "cuda:1"  → second GPU, etc.

    max_image_size: int = 2560
    # Before tiling, if the image's longest side exceeds this value,
    # it is downscaled proportionally. Bboxes are scaled back to
    # original coordinates afterward, so crops are pixel-accurate.
    #
    # WHY THIS MATTERS:
    #   A 6000×4000 image with 640px tiles (step=512) → ~96 tiles → ~28s on CPU.
    #   The same image capped at 2560px → ~16 tiles → ~5s on CPU.
    #   Detection quality is not meaningfully affected — faces that need
    #   sub-pixel precision at 6K resolution are invisible to humans anyway.
    #
    # Set to 0 or a very large number to disable resizing.

    # ── Stage 1 — Detection thresholds ───────────────────────────────────
    det_thresh: float = 0.35
    # Intentionally lower than the default 0.50.
    # Stage 2 validator is the precision gate, so Stage 1 can over-propose.
    # Raising this = fewer candidates reach Stage 2 = higher risk of missing
    # real but borderline faces.

    nms_iou_cross_tile: float = 0.45
    # IoU threshold for NMS that merges results across tiles.
    # Lower = more aggressive suppression (may merge nearby faces).
    # Higher = allows more overlapping boxes (more duplicates).

    # ── Tiling ────────────────────────────────────────────────────────────
    use_tiling: bool = True
    # Enable sliding-window tiling for large images.
    # Prevents small faces being destroyed by downscaling to 640×640.

    tile_size: int = 640
    # Pixel size of each square tile. Matches SCRFD training resolution.

    tile_overlap: float = 0.20
    # Fraction of tile_size used as overlap between adjacent tiles.
    # 0.20 = 20% overlap. Prevents faces at tile edges being cut off.

    tiling_threshold_px: int = 1280
    # If the image is ≤ this value on BOTH sides, tiling is skipped.
    # No benefit in tiling small images; saves compute.

    # ── Stage 2 — Validator ───────────────────────────────────────────────
    min_face_px: int = 15
    # L1: Both bbox width AND height must exceed this in pixels.
    # Eliminates tiny noise: skin textures, background patterns.
    # 15px preserves distant crowd faces.

    aspect_ratio_min: float = 0.5
    # L2: Minimum bbox height/width ratio.
    # A ratio below 0.5 means the box is more than 2× wider than tall —
    # not a face shape.

    aspect_ratio_max: float = 2.4
    # L2: Maximum bbox height/width ratio.
    # Catches arms/hands which produce very elongated boxes (ratio 3–6).
    # 2.4 accommodates open-mouth expressions, tilted heads, and close-ups.

    eye_sep_ratio_min: float = 0.15
    # L3c: Minimum eye separation as a fraction of bbox width.
    # Below this, the "eyes" are basically on top of each other — not a face.

    eye_sep_ratio_max: float = 0.85
    # L3c: Maximum eye separation as a fraction of bbox width.
    # Above this, something is wrong with the landmark placement.

    final_confidence: float = 0.50
    # L4: After geometric validation passes, require this confidence floor.
    # 0.50 filters out blurry, indistinguishable background crowd noise while
    # preserving distinguishable faces, profile views, and expressions.

    final_nms_iou: float = 0.35
    # L5: Stricter IoU for final deduplication after validation.
    # Lower than cross-tile NMS (0.45) — removes residual duplicates.

    # ── Output ────────────────────────────────────────────────────────────
    return_landmarks: bool = True
    # Include 5-point landmark coords in output dicts.
    # Set False if downstream code doesn't need them.

    return_confidence: bool = True
    # Include confidence score in output dicts.
