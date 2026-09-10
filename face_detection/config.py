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
    device: str = "cpu"

    # ── Stage 1 — Detection thresholds ───────────────────────────────────
    det_thresh: float = 0.35
    nms_iou_cross_tile: float = 0.45

    # ── Tiling ────────────────────────────────────────────────────────────
    use_tiling: bool = True
    tile_size: int = 640
    tile_overlap: float = 0.20
    tiling_threshold_px: int = 1280

    # ── Stage 2 — Geometric Validator thresholds ──────────────────────────
    min_face_px: int = 10
    aspect_ratio_min: float = 0.40
    aspect_ratio_max: float = 2.20

    eye_sep_ratio_min: float = 0.15
    eye_sep_ratio_max: float = 0.85

    mouth_eye_ratio_min: float = 0.20
    mouth_eye_ratio_max: float = 2.50

    final_confidence: float = 0.40
    nms_iou_final: float = 0.40

    # ── Output options ────────────────────────────────────────────────────
    return_landmarks: bool = True
    return_confidence: bool = True
