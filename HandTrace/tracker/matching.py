"""
matching.py

Matches hands between consecutive video frames.

Responsibilities:
- Compute distances between hands
- Match previous positions to current-frame hands (one-to-one)
- Report which hands disappeared or appeared

This module DOES NOT:
- Assign IDs
- Track missing hands
- Interpolate
- Smooth landmarks

Those responsibilities belong to tracker.py.

Matching strategy: global greedy by distance. All (previous, current) pairs
within the distance limit are sorted from closest to farthest and assigned
in that order. Unlike matching one previous hand at a time, the result does
not depend on the order of the hands in the list, so two close hands are
less likely to swap IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import TYPE_CHECKING, Sequence

from config import CONFIG
from detector.landmark_utils import compute_centroid_px

if TYPE_CHECKING:  # type hints only; avoids importing mediapipe here
    from detector.hand_detector import HandDetection

Point = tuple[float, float]


@dataclass(frozen=True)
class MatchResult:
    matches: dict[int, int]                  # previous_index -> current_index
    distances: dict[int, float]              # previous_index -> match distance (px)
    unmatched_previous: list[int] = field(default_factory=list)  # disappeared
    unmatched_current: list[int] = field(default_factory=list)   # newly appeared


def euclidean_distance(p1: Point, p2: Point) -> float:
    """Distance between two points."""
    return hypot(p1[0] - p2[0], p1[1] - p2[1])


def match_points(
    previous: Sequence[Point],
    current: Sequence[Point],
    max_distance: float | None = None,
) -> MatchResult:
    """
    Match previous points to current points, one-to-one.

    Works on plain (x, y) points, so the tracker can pass the last known
    position of each track, including tracks whose hand is currently missing.

    Parameters
    ----------
    max_distance:
        Largest allowed distance in pixels. Defaults to
        CONFIG.tracking.max_track_distance.
    """
    if max_distance is None:
        max_distance = CONFIG.tracking.max_track_distance

    # Every pair within the limit, closest first. Index tie-breakers make
    # the result deterministic.
    candidates = sorted(
        (dist, p_idx, c_idx)
        for p_idx, p in enumerate(previous)
        for c_idx, c in enumerate(current)
        if (dist := euclidean_distance(p, c)) <= max_distance
    )

    matches: dict[int, int] = {}
    distances: dict[int, float] = {}
    used_current: set[int] = set()

    for dist, p_idx, c_idx in candidates:
        if p_idx in matches or c_idx in used_current:
            continue
        matches[p_idx] = c_idx
        distances[p_idx] = dist
        used_current.add(c_idx)

    return MatchResult(
        matches=matches,
        distances=distances,
        unmatched_previous=[i for i in range(len(previous)) if i not in matches],
        unmatched_current=[i for i in range(len(current)) if i not in used_current],
    )


def match_hands(
    previous_hands: Sequence[HandDetection],
    current_hands: Sequence[HandDetection],
    width: int,
    height: int,
    max_distance: float | None = None,
) -> MatchResult:
    """Match hands from the previous frame to the current frame by centroid."""
    previous = [compute_centroid_px(h, width, height) for h in previous_hands]
    current = [compute_centroid_px(h, width, height) for h in current_hands]
    return match_points(previous, current, max_distance)
