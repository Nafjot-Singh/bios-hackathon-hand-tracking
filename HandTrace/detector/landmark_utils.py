"""
landmark_utils.py

Pure geometry helpers for hand detections.

No MediaPipe or OpenCV here: these functions only need an object with a
`.landmarks` sequence of items that have `.x` and `.y` (normalized 0..1),
such as `HandDetection` from detector/hand_detector.py.

Usage:
    cx, cy = compute_centroid_px(hand, width, height)
    x0, y0, x1, y1 = compute_bbox_px(hand, width, height)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # type hints only; avoids importing mediapipe here
    from detector.hand_detector import HandDetection

Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max


# -----------------------------
# Normalized (0..1) coordinates
# -----------------------------
def compute_centroid(hand: HandDetection) -> Point:
    """Mean landmark position."""
    n = len(hand.landmarks)
    if n == 0:
        raise ValueError("Hand has no landmarks")
    return (
        sum(lm.x for lm in hand.landmarks) / n,
        sum(lm.y for lm in hand.landmarks) / n,
    )


def compute_bbox(hand: HandDetection) -> BBox:
    """Tight bounding box around all landmarks."""
    if not hand.landmarks:
        raise ValueError("Hand has no landmarks")
    xs = [lm.x for lm in hand.landmarks]
    ys = [lm.y for lm in hand.landmarks]
    return min(xs), min(ys), max(xs), max(ys)


# -----------------------------
# Pixel coordinates
# -----------------------------
def compute_landmarks_px(hand: HandDetection, width: int, height: int) -> list[Point]:
    return [(lm.x * width, lm.y * height) for lm in hand.landmarks]


def compute_centroid_px(hand: HandDetection, width: int, height: int) -> Point:
    cx, cy = compute_centroid(hand)
    return cx * width, cy * height


def compute_bbox_px(hand: HandDetection, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = compute_bbox(hand)
    return x0 * width, y0 * height, x1 * width, y1 * height
