"""
scrfd.py — Thin wrapper around InsightFace's SCRFD face detector.

Why wrap it?
  - Isolates the InsightFace import from the rest of the module.
  - If you ever need to swap the model (e.g., to SCRFD-34GF or a custom
    ONNX model), you only edit this file. Nothing else changes.
  - Normalises output to a plain list of dicts so postprocess/validator
    have no dependency on InsightFace's internal Face objects.

Model: buffalo_l (InsightFace pack)
  Contains SCRFD-10GF ONNX weights + ArcFace recognition model.
  We load detection-only via allowed_modules=["detection"] to skip the
  recognition model entirely — saves ~200 MB RAM and startup time.

First run: InsightFace automatically downloads buffalo_l (~300 MB)
  to ~/.insightface/models/buffalo_l/. Subsequent runs use the cache.
"""

import numpy as np
from face_detection.config import DetectorConfig


class SCRFDDetector:
    """Wraps InsightFace FaceAnalysis for detection-only inference."""

    def __init__(self, cfg: DetectorConfig):
        """
        Load and prepare the SCRFD model.

        Args:
            cfg: DetectorConfig instance.

        The model is prepared once here and reused across all detect() calls.
        This is intentional — model loading takes ~1–3 s; inference is fast.
        """
        # ctx_id: -1 = CPU, 0 = first GPU, 1 = second GPU, etc.
        if cfg.device == "cpu":
            ctx_id = -1
        else:
            try:
                ctx_id = int(cfg.device.split(":")[-1])
            except (ValueError, IndexError):
                raise ValueError(
                    f"Invalid device string: '{cfg.device}'. "
                    "Use 'cpu' or 'cuda:N' (e.g., 'cuda:0')."
                )

        try:
            from insightface.app import FaceAnalysis
        except ImportError as e:
            raise ImportError(
                "InsightFace is not installed. Run:\n"
                "  pip install insightface onnxruntime\n"
                "  (or onnxruntime-gpu if you have CUDA)"
            ) from e

        self._app = FaceAnalysis(
            name=cfg.model_name,
            allowed_modules=["detection"],  # skip recognition — saves RAM
        )
        self._app.prepare(
            ctx_id=ctx_id,
            det_thresh=cfg.det_thresh,
            det_size=(640, 640),
        )

    def detect(self, image: np.ndarray) -> list:
        """
        Run face detection on a single image (or tile).

        Args:
            image: BGR numpy array (H, W, 3), uint8.

        Returns:
            List of dicts:
            {
                "bbox":       [x1, y1, x2, y2],  # int pixel coords
                "confidence": float,              # model score 0.0–1.0
                "landmarks":  [[x,y], ...] | None # 5 keypoints, or None
            }
        """
        faces = self._app.get(image)
        results = []

        for face in faces:
            bbox = face.bbox.astype(int).tolist()   # [x1, y1, x2, y2]
            conf = float(face.det_score)

            # kps = keypoints: left_eye, right_eye, nose, mouth_l, mouth_r
            kps = face.kps.tolist() if face.kps is not None else None

            results.append({
                "bbox":       bbox,
                "confidence": conf,
                "landmarks":  kps,
            })

        return results
