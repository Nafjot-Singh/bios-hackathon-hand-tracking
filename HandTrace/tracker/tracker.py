"""
tracker.py

Persistent hand tracking across video frames.

Responsibilities:
- Assign persistent IDs (1, 2, 3, ...) to hands
- Keep a track alive for a limited number of frames when its hand disappears
- Re-attach a returning hand to its old ID if it comes back before the timeout
- Delete tracks that stay missing for too long

This module DOES NOT:
- Smooth landmarks
- Interpolate missing hands
- Write JSON or visualize anything

Frame-to-frame matching is delegated to matching.py.

Usage:
    tracker = HandTracker()

    for frame in reader.frames():
        hands = detector.detect(frame.image, frame.timestamp_ms)
        tracks = tracker.update(hands, reader.width, reader.height)

        for track in tracker.visible_tracks:
            ...  # track.hand holds this frame's detection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from config import CONFIG
from detector.landmark_utils import compute_centroid_px
from tracker.matching import match_hands

if TYPE_CHECKING:  # type hints only; keeps this module free of mediapipe
    from detector.hand_detector import HandDetection

Point = tuple[float, float]


# ---------------------------------------------------------
# Data classes
# ---------------------------------------------------------

@dataclass(slots=True)
class Track:
    """
    A hand followed over time.

    Attributes
    ----------
    track_id:
        Persistent ID, unique among all tracks the tracker has created.
    hand:
        Most recent detection for this track. While the hand is missing,
        this is the last detection seen (it is NOT refreshed or predicted).
    centroid:
        Centroid of `hand` in pixel coordinates.
    missing_frames:
        Consecutive frames without a matching detection (0 when visible).
    age:
        Number of frames since the track was created, counting the frame
        it was created in (so a new track has age 1).
    updated_this_frame:
        True if a detection was matched to this track in the latest update.
    """

    track_id: int
    hand: HandDetection
    centroid: Point
    missing_frames: int = 0
    age: int = 1
    updated_this_frame: bool = True


# ---------------------------------------------------------
# Tracker
# ---------------------------------------------------------

class HandTracker:
    """
    Keeps hand IDs stable from frame to frame.

    A track whose hand disappears stays alive for `max_missing_frames`
    frames. If a hand reappears near its last known position within that
    time, it gets the same ID back. Otherwise the track is deleted, and a
    hand appearing later receives a brand-new ID: IDs are never reused.
    """

    def __init__(
        self,
        max_track_distance: float | None = None,
        max_missing_frames: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        max_track_distance:
            Largest pixel distance a hand may move between frames and still
            keep its ID. Defaults to CONFIG.tracking.max_track_distance.
        max_missing_frames:
            Frames a track survives without a detection. Defaults to
            CONFIG.tracking.max_missing_frames.
        """
        self._max_track_distance: float = (
            CONFIG.tracking.max_track_distance
            if max_track_distance is None
            else max_track_distance
        )
        self._max_missing_frames: int = (
            CONFIG.tracking.max_missing_frames
            if max_missing_frames is None
            else max_missing_frames
        )

        if self._max_track_distance <= 0:
            raise ValueError("max_track_distance must be > 0")
        if self._max_missing_frames < 0:
            raise ValueError("max_missing_frames must be >= 0")

        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    @property
    def active_tracks(self) -> list[Track]:
        """
        All live tracks (visible or temporarily missing), sorted by ID.

        The list is new on every call, but the Track objects are the
        tracker's own and are updated in place by the next `update()`.
        To keep a record of past frames, copy the values you need
        (e.g. `dataclasses.replace(track)`) instead of storing the objects.
        """
        # Sorted by construction: IDs only ever increase and new tracks are
        # appended, while deletions never reorder the dict.
        return list(self._tracks.values())

    @property
    def visible_tracks(self) -> list[Track]:
        """
        Tracks matched to a detection in the latest update, sorted by ID.

        Use this for drawing, JSON output and gesture recognition so they
        don't each need to check `updated_this_frame`.
        """
        return [track for track in self._tracks.values() if track.updated_this_frame]

    def reset(self) -> None:
        """Remove all tracks and restart ID numbering from 1."""
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        hands: Sequence[HandDetection],
        width: int,
        height: int,
    ) -> list[Track]:
        """
        Process one frame of detections.

        Parameters
        ----------
        hands:
            Detections for the current frame.
        width, height:
            Frame size in pixels (used to convert centroids to pixels).

        Returns
        -------
        list[Track]
            Active tracks sorted by ID. Tracks whose hand is currently
            missing but not yet expired are included, with
            `updated_this_frame == False`. Use `visible_tracks` for only
            the hands detected in this frame.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame size: {width}x{height}")

        # Match indices refer to positions in this list (ID order).
        previous_tracks = list(self._tracks.values())

        # Steps 1-2: compare each track's last known hand with current hands.
        result = match_hands(
            [track.hand for track in previous_tracks],
            hands,
            width,
            height,
            max_distance=self._max_track_distance,
        )

        # Step 3: refresh matched tracks.
        for prev_idx, curr_idx in result.matches.items():
            self._refresh_track(previous_tracks[prev_idx], hands[curr_idx], width, height)

        # Step 4: age tracks that found no hand this frame.
        for prev_idx in result.unmatched_previous:
            self._mark_missing(previous_tracks[prev_idx])

        # Step 5: delete tracks that have been gone too long.
        self._remove_expired()

        # Step 6: start new tracks for hands nobody claimed.
        for curr_idx in result.unmatched_current:
            self._create_track(hands[curr_idx], width, height)

        # Step 7: report.
        return self.active_tracks

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    @staticmethod
    def _refresh_track(
        track: Track,
        hand: HandDetection,
        width: int,
        height: int,
    ) -> None:
        """Attach a new detection to an existing track."""
        track.hand = hand
        track.centroid = compute_centroid_px(hand, width, height)
        track.missing_frames = 0
        track.age += 1
        track.updated_this_frame = True

    @staticmethod
    def _mark_missing(track: Track) -> None:
        """Record that a track had no detection this frame."""
        track.missing_frames += 1
        track.age += 1
        track.updated_this_frame = False

    def _remove_expired(self) -> None:
        """Delete tracks missing for more than `max_missing_frames` frames."""
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missing_frames > self._max_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _create_track(self, hand: HandDetection, width: int, height: int) -> Track:
        """Start a new track with the next unused ID."""
        track = Track(
            track_id=self._next_id,
            hand=hand,
            centroid=compute_centroid_px(hand, width, height),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1  # monotonic counter: IDs are never reused
        return track
