"""
postprocess.py — NMS and deduplication utilities.

Pure Python / numpy. No external dependencies beyond numpy.
Used for:
  1. Cross-tile NMS: merge results from overlapping image tiles.
  2. Final deduplication NMS: after validation, remove residual duplicates.
"""

from typing import List


def iou(box_a: list, box_b: list) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2]
        box_b: [x1, y1, x2, y2]

    Returns:
        IoU value in [0.0, 1.0]
    """
    # Intersection rectangle
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def nms(detections: List[dict], iou_threshold: float) -> List[dict]:
    """
    Standard Non-Maximum Suppression (greedy).

    Sorts detections by confidence descending. Keeps the highest-confidence
    box and removes any box that overlaps it beyond iou_threshold.
    Repeats until no candidates remain.

    Args:
        detections: list of dicts, each must have 'bbox' and 'confidence'.
        iou_threshold: boxes with IoU > this value are suppressed.

    Returns:
        Filtered list of detections (highest confidence kept per cluster).
    """
    if not detections:
        return []

    # Sort highest confidence first
    candidates = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept: List[dict] = []

    while candidates:
        best = candidates.pop(0)
        kept.append(best)
        # Remove all candidates that overlap too much with the chosen best box
        candidates = [
            d for d in candidates
            if iou(best["bbox"], d["bbox"]) < iou_threshold
        ]

    return kept
