"""
loader.py — Normalize any image input to a BGR numpy array.

Accepted input types:
  - str / pathlib.Path  : file path on disk
  - numpy.ndarray       : already-loaded image (BGR, RGB, RGBA, or grayscale)
  - PIL.Image.Image     : PIL/Pillow image object

All outputs are uint8 BGR arrays (what OpenCV and InsightFace expect).
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


def load_image(source) -> np.ndarray:
    """
    Load an image from any supported source and return a BGR numpy array.

    Args:
        source: file path (str/Path), numpy array, or PIL Image.

    Returns:
        np.ndarray of shape (H, W, 3), dtype uint8, channel order BGR.

    Raises:
        FileNotFoundError: if a path is given and the file cannot be opened.
        ValueError: if the source type is not supported.
    """
    # ── File path ──────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(
                f"Could not read image from path: {source}\n"
                "Check the file exists and is a supported image format."
            )
        return img  # cv2.imread always returns BGR

    # ── NumPy array ────────────────────────────────────────────────────────
    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            # Grayscale (H, W) → BGR (H, W, 3)
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

        if source.ndim == 3:
            channels = source.shape[2]
            if channels == 4:
                # BGRA or RGBA → BGR
                # Assume BGRA (OpenCV convention); if you pass RGB arrays,
                # convert before calling this function or set channels == 3.
                return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            if channels == 3:
                # Return a copy so callers can't accidentally mutate the original
                return source.copy()

        raise ValueError(
            f"Unsupported numpy array shape: {source.shape}. "
            "Expected (H,W), (H,W,3), or (H,W,4)."
        )

    # ── PIL Image ──────────────────────────────────────────────────────────
    elif _PIL_AVAILABLE and isinstance(source, PILImage.Image):
        # Convert to RGB first (handles palette, RGBA, L, etc.)
        rgb = np.array(source.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── Unsupported ────────────────────────────────────────────────────────
    else:
        supported = "str, Path, numpy.ndarray, PIL.Image"
        if not _PIL_AVAILABLE:
            supported = "str, Path, numpy.ndarray  (install Pillow for PIL support)"
        raise ValueError(
            f"Unsupported image source type: {type(source).__name__}. "
            f"Supported types: {supported}"
        )
