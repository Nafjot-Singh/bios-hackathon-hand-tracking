"""
smoother.py

Smooths hand landmarks over time with an exponential moving average.

Responsibilities:
- Keep the previous smoothed landmarks of every track
- Blend each new detection with that history to reduce jitter
- Forget a track's history when it has been missing too long or has expired

Landmark and HandDetection are frozen dataclasses, so they are never edited
in place. For each visible track a NEW HandDetection is built and assigned to
`track.hand` (Track itself is mutable).

This module DOES NOT:
- Detect hands
- Assign track IDs
- Interpolate missing hands
- Write JSON or visualize anything

Usage:
    smoother = LandmarkSmoother(alpha=CONFIG.smoothing.alpha,
                                max_missed=CONFIG.tracking.max_missing_frames)

    tracks = tracker.update(hands, width, height)
    smoother.smooth_tracks(tracks)      # pass ALL active tracks
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # type hints only
    from tracker.tracker import Track

Coords = tuple[float, float, float]


class LandmarkSmoother:
    """
    Exponential moving average over landmark positions, per track.

        smoothed = alpha * new + (1 - alpha) * previous_smoothed

    alpha = 1.0 means no smoothing; smaller values are smoother but lag more.
    """

    def __init__(self, alpha: float = 0.5, max_missed: int = 5) -> None:
        """
        Parameters
        ----------
        alpha:
            Weight of the newest detection, in (0, 1].
        max_missed:
            If a track is missing for more than this many consecutive frames,
            its history is dropped, so when the hand returns it starts fresh
            instead of being blended with an old pose.
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be >= 0")

        self._alpha = alpha
        self._max_missed = max_missed
        self._history: dict[int, tuple[Coords, ...]] = {}  # track_id -> smoothed landmarks

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def smooth_tracks(self, tracks: Iterable[Track]) -> None:
        """
        Smooth the landmarks of every visible track (in place on the Track).

        Pass ALL active tracks, not only visible ones: tracks that are
        missing keep their last pose, and any track that is no longer in
        the list has expired, so its history is removed.
        """
        active = list(tracks)

        # Forget tracks the tracker has deleted.
        active_ids = {track.track_id for track in active}
        for track_id in list(self._history):
            if track_id not in active_ids:
                del self._history[track_id]

        for track in active:
            if not track.updated_this_frame:
                # Coasting: keep the last pose, but don't keep stale history forever.
                if track.missing_frames > self._max_missed:
                    self._history.pop(track.track_id, None)
                continue

            self._smooth_track(track)

    def reset(self) -> None:
        """Forget all history."""
        self._history.clear()

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    def _smooth_track(self, track: Track) -> None:
        """Blend one visible track with its history and store the result."""
        landmarks = track.hand.landmarks
        raw: tuple[Coords, ...] = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
        previous = self._history.get(track.track_id)

        # First sighting (or history dropped), or an unexpected landmark count:
        # start from the raw detection.
        if previous is None or len(previous) != len(raw):
            smoothed = raw
        else:
            a = self._alpha
            smoothed = tuple(
                (
                    a * x + (1 - a) * px,
                    a * y + (1 - a) * py,
                    a * z + (1 - a) * pz,
                )
                for (x, y, z), (px, py, pz) in zip(raw, previous)
            )

        self._history[track.track_id] = smoothed

        # Build new frozen objects instead of editing the old ones.
        landmark_type = type(landmarks[0]) if landmarks else None
        if landmark_type is not None:
            new_landmarks = tuple(landmark_type(x, y, z) for x, y, z in smoothed)
            track.hand = replace(track.hand, landmarks=new_landmarks)
