"""
cues.py

Per-frame hand-object contact decision from several cues.

Bounding-box overlap alone is NEVER contact (Hard Rule 6). A fingertip must be
close to the object, and a second cue (palm proximity or overlap) must agree.
The cues are then fused into one score:

    tip     fingertip-to-box distance / hand scale  (required)
    palm    palm-center-to-box distance / hand scale
    overlap share of the hand box that lies inside the object box

Distances are divided by the hand scale (wrist to middle-finger base, in
pixels) so thresholds work at any resolution.

Result states: nothing, portable_object, fixed_surface or unknown (with a
reason). Temporal smoothing happens later in state_machine.py.

This module DOES NOT detect hands or objects and does not track anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING, Sequence

from config import CONFIG, Contact
from detector.landmark_utils import compute_bbox_px, compute_landmarks_px

if TYPE_CHECKING:  # type hints only
    from detector.hand_detector import HandDetection
    from objects.object_detector import ObjectDetection

Point = tuple[float, float]
Box = tuple[float, float, float, float]

WRIST, MIDDLE_MCP = 0, 9
FINGERTIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class FrameContact:
    """Contact decision for one hand in one frame."""

    state: str                          # nothing | portable_object | fixed_surface | unknown
    obj: ObjectDetection | None = None  # the touched object (contact states only)
    reason: str | None = None           # why unknown: low_quality | conflicting_cues
    score: float = 0.0                  # fused score of the chosen object
    tracking_quality: float = 0.0       # hand tracking quality score (TQS)


# ----------------------------------------------------
# Geometry helpers
# ----------------------------------------------------

def point_to_box_distance(point: Point, box: Box) -> float:
    """Distance from a point to a box (0 if the point is inside)."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return hypot(dx, dy)


def overlap_share(hand_box: Box, object_box: Box) -> float:
    """Part of the hand box (0..1) that lies inside the object box."""
    ix = max(0.0, min(hand_box[2], object_box[2]) - max(hand_box[0], object_box[0]))
    iy = max(0.0, min(hand_box[3], object_box[3]) - max(hand_box[1], object_box[1]))
    hand_area = max(0.0, hand_box[2] - hand_box[0]) * max(0.0, hand_box[3] - hand_box[1])
    return (ix * iy) / hand_area if hand_area > 0 else 0.0


def tracking_quality(hand: HandDetection) -> float:
    """
    Tracking quality score (TQS) in 0..1: handedness confidence times the
    share of landmarks inside the image. A hand cut off by the frame edge,
    or detected with low confidence, scores low.
    """
    landmarks = hand.landmarks
    if not landmarks:
        return 0.0
    inside = sum(0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0 for lm in landmarks)
    return max(0.0, min(1.0, hand.score)) * inside / len(landmarks)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ----------------------------------------------------
# Decision
# ----------------------------------------------------

def classify_frame(
    hand: HandDetection,
    objects: Sequence[ObjectDetection],
    width: int,
    height: int,
    settings: Contact = CONFIG.contact,
) -> FrameContact:
    """
    Decide what one hand touches in one frame.

    Priority when several objects qualify: a portable object beats a fixed
    surface (holding a tool above a bench is a grasp, not a surface contact).
    """
    tqs = tracking_quality(hand)
    if tqs < settings.min_tracking_quality:
        return FrameContact("unknown", reason="low_quality", tracking_quality=tqs)

    points = compute_landmarks_px(hand, width, height)
    scale = hypot(points[WRIST][0] - points[MIDDLE_MCP][0], points[WRIST][1] - points[MIDDLE_MCP][1])
    if scale < 1.0:  # degenerate hand (all landmarks on one spot)
        return FrameContact("unknown", reason="low_quality", tracking_quality=tqs)

    tips = [points[i] for i in FINGERTIPS]
    palm = (
        sum(points[i][0] for i in PALM_POINTS) / len(PALM_POINTS),
        sum(points[i][1] for i in PALM_POINTS) / len(PALM_POINTS),
    )
    hand_box = compute_bbox_px(hand, width, height)
    frame_area = float(width * height)

    best: dict[str, tuple[float, ObjectDetection]] = {}  # kind -> (score, object)
    rejected_large_surface = False

    for obj in objects:
        d_tip = min(point_to_box_distance(tip, obj.bbox) for tip in tips) / scale
        d_palm = point_to_box_distance(palm, obj.bbox) / scale
        overlap = overlap_share(hand_box, obj.bbox)

        # Gate: a fingertip must be near AND a second cue must agree.
        if d_tip > settings.tip_distance_max:
            continue
        if d_palm > settings.palm_distance_max and overlap < settings.min_overlap:
            continue

        score = (
            settings.weight_tip * _clamp01(1.0 - d_tip / settings.tip_distance_max)
            + settings.weight_palm * _clamp01(1.0 - d_palm / settings.palm_distance_max)
            + settings.weight_overlap * _clamp01(overlap / 0.5)
        )
        if score < settings.score_threshold:
            continue

        # A "fixed surface" that fills the frame is a detector artifact, not a surface to touch.
        if obj.kind == "fixed" and obj.area / frame_area > settings.max_fixed_surface_area_ratio:
            rejected_large_surface = True
            continue

        if obj.kind not in best or score > best[obj.kind][0]:
            best[obj.kind] = (score, obj)

    if "portable" in best:
        score, obj = best["portable"]
        return FrameContact("portable_object", obj, None, score, tqs)
    if "fixed" in best:
        score, obj = best["fixed"]
        return FrameContact("fixed_surface", obj, None, score, tqs)
    if rejected_large_surface:
        return FrameContact("unknown", reason="conflicting_cues", tracking_quality=tqs)
    return FrameContact("nothing", tracking_quality=tqs)
