"""
metrics.py

Computes tracking statistics from the JSON Lines output.

Input:
    output/hands.jsonl

Output:
    A dictionary (or TrackingMetrics object) with tracking metrics, and
    optionally output/metrics.json.

This module DOES NOT:
- Draw graphs
- Generate reports
- Modify tracking results

Those responsibilities belong to graphs.py and report.py.

Important: hands.jsonl only contains frames where a hand was visible, so the
true number of video frames cannot be recovered from it. Pass `total_frames`
(e.g. from VideoReader or the Stats object in main.py) to get an exact
detection rate. Without it, the last frame with a hand is used, which
overestimates the rate when the video ends without hands.

ID switches are ESTIMATED without ground truth: see `_find_id_switches`.

Usage:
    calculator = MetricsCalculator(total_frames=1200)
    calculator.print_summary()
    calculator.save()                         # output/metrics.json
    metrics = calculator.calculate()          # plain dict
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator

from config import CONFIG

log = logging.getLogger(__name__)

_REQUIRED_KEYS = ("frame", "track_id", "confidence")


# ----------------------------------------------------
# Result containers
# ----------------------------------------------------

@dataclass(frozen=True)
class TrackSummary:
    """Lifetime of one track."""

    track_id: int
    handedness: str          # most common label seen on this track
    first_frame: int
    last_frame: int
    visible_frames: int      # frames where the track had a detection
    span: int                # first to last frame, inclusive
    gap_frames: int          # frames missing inside the span
    longest_gap: int         # longest run of consecutive missing frames


@dataclass(frozen=True)
class TrackingMetrics:
    """All metrics computed from one hands.jsonl file."""

    # Frames
    total_frames: int
    total_frames_is_estimate: bool       # True when total_frames was not given
    frames_with_hands: int
    frames_without_hands: int
    detection_rate: float                # frames_with_hands / total_frames

    # Records
    records: int
    skipped_lines: int                   # unreadable / incomplete lines

    # Hands per frame
    average_hands_per_frame: float       # over frames that contain hands
    maximum_hands_in_frame: int
    hands_per_frame_histogram: dict[int, int] = field(default_factory=dict)

    # Confidence
    average_confidence: float = 0.0
    median_confidence: float = 0.0
    minimum_confidence: float = 0.0
    maximum_confidence: float = 0.0
    low_confidence_records: int = 0      # confidence below the threshold
    low_confidence_threshold: float = 0.0

    # Tracks
    unique_tracks: int = 0
    average_track_length: float = 0.0    # frames where the track was visible
    longest_track: int = 0
    shortest_track: int = 0
    average_track_span: float = 0.0      # first to last visible frame
    tracks_with_gaps: int = 0            # tracks that vanished and came back
    total_gap_frames: int = 0            # frames missing inside track spans

    # Disappearances (a hand missing, then back under the SAME ID)
    disappearance_events: int = 0
    longest_disappearance: int = 0       # frames
    longest_disappearance_track: int | None = None

    # ID switches (estimated)
    id_switches: int = 0
    switch_window_frames: int = 0
    id_switch_events: list[dict[str, Any]] = field(default_factory=list)

    # Handedness
    handedness_counts: dict[str, int] = field(default_factory=dict)

    # Per-track lifetimes (for graphs and reports)
    tracks: list[TrackSummary] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the metrics as a plain dictionary."""
        return asdict(self)


# ----------------------------------------------------
# Calculator
# ----------------------------------------------------

class MetricsCalculator:
    """Calculates statistics from hands.jsonl."""

    def __init__(
        self,
        json_path: Path | str | None = None,
        total_frames: int | None = None,
        low_confidence_threshold: float = 0.8,
        switch_window_frames: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        json_path:
            Path to the JSONL file. Defaults to CONFIG.paths.output_json.
        total_frames:
            Number of frames in the video. Strongly recommended: see the
            module docstring.
        low_confidence_threshold:
            Records with confidence below this value are counted as
            low-confidence.
        switch_window_frames:
            Largest gap (in frames) between one track ending and a new
            track of the same handedness starting for it to count as an
            ID switch. Roughly one second at 30 FPS by default.
        """
        self.json_path = Path(json_path or CONFIG.paths.output_json)

        if not self.json_path.is_file():
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        if total_frames is not None and total_frames < 0:
            raise ValueError("total_frames must be >= 0")
        if not 0.0 <= low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be between 0 and 1")
        if switch_window_frames < 0:
            raise ValueError("switch_window_frames must be >= 0")

        self.total_frames = total_frames
        self.low_confidence_threshold = low_confidence_threshold
        self.switch_window_frames = switch_window_frames

        self._cache: TrackingMetrics | None = None
        self._skipped = 0

    # ----------------------------------------------------
    # Reading
    # ----------------------------------------------------

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        """
        Yield valid records one at a time (the file is streamed, so large
        files with many landmarks do not need to fit in memory).

        Blank lines are ignored. Lines that are not valid JSON (for example
        a last line cut off by a crash) or lack required keys are skipped
        and counted in `self._skipped`.
        """
        with self.json_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Skipping invalid JSON on line %d", line_number)
                    self._skipped += 1
                    continue

                if not isinstance(record, dict) or any(k not in record for k in _REQUIRED_KEYS):
                    log.warning("Skipping incomplete record on line %d", line_number)
                    self._skipped += 1
                    continue

                yield record

    # ----------------------------------------------------
    # Metrics
    # ----------------------------------------------------

    def calculate(self) -> dict[str, Any]:
        """Return all tracking statistics as a dictionary."""
        return self.compute().to_dict()

    def compute(self) -> TrackingMetrics:
        """Return all tracking statistics as a TrackingMetrics object."""
        if self._cache is None:
            self._cache = self._compute()
        return self._cache

    def _compute(self) -> TrackingMetrics:
        self._skipped = 0

        hands_in_frame: Counter[int] = Counter()
        track_frames: defaultdict[int, list[int]] = defaultdict(list)
        track_handedness: defaultdict[int, Counter[str]] = defaultdict(Counter)
        handedness: Counter[str] = Counter()
        confidences: list[float] = []

        for record in self._iter_records():
            frame = record["frame"]
            track = record["track_id"]
            label = record.get("handedness", "Unknown")

            hands_in_frame[frame] += 1
            track_frames[track].append(frame)
            track_handedness[track][label] += 1
            handedness[label] += 1
            confidences.append(record["confidence"])

        frames_with_hands = len(hands_in_frame)
        last_frame_seen = max(hands_in_frame, default=-1)

        # Total frames: exact if provided, otherwise estimated.
        if self.total_frames is None:
            total_frames = last_frame_seen + 1
            is_estimate = True
        else:
            if last_frame_seen > self.total_frames:
                raise ValueError(
                    f"total_frames={self.total_frames} is smaller than the "
                    f"highest frame in the file ({last_frame_seen})"
                )
            # Use whichever is larger: the passed-in count or the 0-based
            # index + 1 from the file (handles early Ctrl+C or off-by-one).
            total_frames = max(self.total_frames, last_frame_seen + 1)
            is_estimate = False

        summaries, disappearance_events = self._summarize_tracks(
            track_frames, track_handedness
        )
        switches = self._find_id_switches(summaries, self.switch_window_frames)

        longest = max(summaries, key=lambda s: s.longest_gap, default=None)
        visible = [s.visible_frames for s in summaries]
        hands_per_frame = list(hands_in_frame.values())

        return TrackingMetrics(
            total_frames=total_frames,
            total_frames_is_estimate=is_estimate,
            frames_with_hands=frames_with_hands,
            frames_without_hands=total_frames - frames_with_hands,
            detection_rate=frames_with_hands / total_frames if total_frames else 0.0,
            records=len(confidences),
            skipped_lines=self._skipped,
            average_hands_per_frame=mean(hands_per_frame) if hands_per_frame else 0.0,
            maximum_hands_in_frame=max(hands_per_frame, default=0),
            hands_per_frame_histogram=dict(sorted(Counter(hands_per_frame).items())),
            average_confidence=mean(confidences) if confidences else 0.0,
            median_confidence=median(confidences) if confidences else 0.0,
            minimum_confidence=min(confidences, default=0.0),
            maximum_confidence=max(confidences, default=0.0),
            low_confidence_records=sum(
                c < self.low_confidence_threshold for c in confidences
            ),
            low_confidence_threshold=self.low_confidence_threshold,
            unique_tracks=len(summaries),
            average_track_length=mean(visible) if visible else 0.0,
            longest_track=max(visible, default=0),
            shortest_track=min(visible, default=0),
            average_track_span=mean(s.span for s in summaries) if summaries else 0.0,
            tracks_with_gaps=sum(s.gap_frames > 0 for s in summaries),
            total_gap_frames=sum(s.gap_frames for s in summaries),
            disappearance_events=disappearance_events,
            longest_disappearance=longest.longest_gap if longest else 0,
            longest_disappearance_track=(
                longest.track_id if longest and longest.longest_gap > 0 else None
            ),
            id_switches=len(switches),
            switch_window_frames=self.switch_window_frames,
            id_switch_events=switches,
            handedness_counts=dict(sorted(handedness.items())),
            tracks=summaries,
        )

    @staticmethod
    def _summarize_tracks(
        track_frames: dict[int, list[int]],
        track_handedness: dict[int, Counter[str]],
    ) -> tuple[list[TrackSummary], int]:
        """
        Build one TrackSummary per track.

        Returns the summaries (sorted by track ID) and the total number of
        disappearance events (gaps inside tracks).
        """
        summaries: list[TrackSummary] = []
        disappearance_events = 0

        for track_id in sorted(track_frames):
            frames = sorted(set(track_frames[track_id]))

            # Runs of missing frames between consecutive visible frames.
            gaps = [b - a - 1 for a, b in zip(frames, frames[1:]) if b - a > 1]
            disappearance_events += len(gaps)

            summaries.append(
                TrackSummary(
                    track_id=track_id,
                    handedness=track_handedness[track_id].most_common(1)[0][0],
                    first_frame=frames[0],
                    last_frame=frames[-1],
                    visible_frames=len(frames),
                    span=frames[-1] - frames[0] + 1,
                    gap_frames=sum(gaps),
                    longest_gap=max(gaps, default=0),
                )
            )

        return summaries, disappearance_events

    @staticmethod
    def _find_id_switches(
        summaries: list[TrackSummary],
        window: int,
    ) -> list[dict[str, Any]]:
        """
        Estimate ID switches.

        Without ground truth we cannot be sure which hand is which, so an
        ID switch is counted when:
          - a track ends, and
          - a NEW track with the same handedness starts within `window`
            frames, without overlapping the old track.

        Example: a Left hand is tracked as ID 1 until frame 19, then as ID 5
        from frame 25. That is one switch (ID 1 -> ID 5, 5 frames apart).

        Each ended track can be the predecessor of only one new track. A
        hand that really left the scene and came back after more than
        `window` frames is not counted. Two same-handed tracks alive at the
        same time are treated as two different hands, not a switch.
        """
        by_label: defaultdict[str, list[TrackSummary]] = defaultdict(list)
        for summary in summaries:
            by_label[summary.handedness].append(summary)

        events: list[dict[str, Any]] = []

        for label, group in by_label.items():
            group.sort(key=lambda s: (s.first_frame, s.track_id))
            claimed: set[int] = set()  # predecessors already matched

            for i, new in enumerate(group):
                candidates = [
                    old for old in group[:i]
                    if old.track_id not in claimed
                    and old.last_frame < new.first_frame
                    and new.first_frame - old.last_frame <= window
                ]
                if not candidates:
                    continue

                old = max(candidates, key=lambda s: s.last_frame)  # closest in time
                claimed.add(old.track_id)
                events.append(
                    {
                        "handedness": label,
                        "from_track": old.track_id,
                        "to_track": new.track_id,
                        "last_frame_old": old.last_frame,
                        "first_frame_new": new.first_frame,
                        "gap_frames": new.first_frame - old.last_frame - 1,
                    }
                )

        events.sort(key=lambda e: (e["first_frame_new"], e["to_track"]))
        return events

    # ----------------------------------------------------
    # Saving
    # ----------------------------------------------------

    def save(self, output_path: Path | str | None = None) -> Path:
        """
        Save the metrics as JSON.

        Parameters
        ----------
        output_path:
            Destination file. Defaults to CONFIG.paths.output_metrics.

        Returns
        -------
        Path
            The file that was written.
        """
        path = Path(output_path or CONFIG.paths.output_metrics)
        path.parent.mkdir(parents=True, exist_ok=True)

        text = json.dumps(self.calculate(), indent=CONFIG.json_indent, allow_nan=False)

        # Write to a temp file first so a failure never leaves a half-written file.
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(text + "\n", encoding="utf-8")
        temp.replace(path)
        return path

    # ----------------------------------------------------
    # Printing
    # ----------------------------------------------------

    def print_summary(self) -> None:
        """Print a readable summary of the metrics."""
        m = self.compute()

        def row(label: str, value: object) -> None:
            print(f"  {label:<26}: {value}")

        estimate = " (estimated)" if m.total_frames_is_estimate else ""

        print("\n========== Tracking Metrics ==========")

        print("\nFrames")
        row("Total frames", f"{m.total_frames}{estimate}")
        row("With hands", m.frames_with_hands)
        row("Without hands", m.frames_without_hands)
        row("Detection rate", f"{m.detection_rate:.1%}")

        print("\nHands")
        row("Records", m.records)
        row("Avg hands / frame", f"{m.average_hands_per_frame:.2f}")
        row("Max hands in a frame", m.maximum_hands_in_frame)
        row(
            "Hands per frame",
            ", ".join(f"{n} hand(s): {c}" for n, c in m.hands_per_frame_histogram.items())
            or "-",
        )
        row(
            "Handedness",
            ", ".join(f"{k}: {v}" for k, v in m.handedness_counts.items()) or "-",
        )

        print("\nConfidence")
        row("Average / median", f"{m.average_confidence:.3f} / {m.median_confidence:.3f}")
        row("Min / max", f"{m.minimum_confidence:.3f} / {m.maximum_confidence:.3f}")
        row(f"Below {m.low_confidence_threshold:g}", m.low_confidence_records)

        print("\nTracks")
        row("Unique tracks", m.unique_tracks)
        row("Avg length (visible)", f"{m.average_track_length:.1f} frames")
        row("Avg span (first-last)", f"{m.average_track_span:.1f} frames")
        row("Longest / shortest", f"{m.longest_track} / {m.shortest_track} frames")

        print("\nStability")
        row("Tracks with gaps", m.tracks_with_gaps)
        row("Disappearances", m.disappearance_events)
        row("Gap frames (total)", m.total_gap_frames)
        longest = f"{m.longest_disappearance} frames"
        if m.longest_disappearance_track is not None:
            longest += f" (track {m.longest_disappearance_track})"
        row("Longest disappearance", longest)
        row(f"ID switches (<= {m.switch_window_frames} f)", m.id_switches)
        for event in m.id_switch_events[:10]:
            print(
                f"      {event['handedness']}: ID {event['from_track']} -> "
                f"ID {event['to_track']} "
                f"(frames {event['last_frame_old']} -> {event['first_frame_new']})"
            )
        if len(m.id_switch_events) > 10:
            print(f"      ... and {len(m.id_switch_events) - 10} more")

        self.print_track_lifetimes()

        if m.skipped_lines:
            print(f"\n  WARNING: {m.skipped_lines} unreadable line(s) were skipped.")
        if m.total_frames_is_estimate:
            print("\n  Note: pass total_frames for an exact detection rate.")

        print("\n======================================\n")

    def print_track_lifetimes(self, max_tracks: int = 20, width: int = 40) -> None:
        """
        Print a text bar chart of how long each track was visible.

        Parameters
        ----------
        max_tracks:
            Show at most this many tracks (in ID order).
        width:
            Bar length for the longest track, in characters.
        """
        m = self.compute()
        if not m.tracks:
            return

        longest = max(s.visible_frames for s in m.tracks)

        print("\nTrack lifetimes (visible frames)")
        for summary in m.tracks[:max_tracks]:
            bar = "█" * max(1, round(width * summary.visible_frames / longest))
            print(
                f"  Track {summary.track_id:>3} {summary.handedness[:5]:<5} "
                f"{bar} {summary.visible_frames}"
            )
        if len(m.tracks) > max_tracks:
            print(f"  ... and {len(m.tracks) - max_tracks} more tracks")


# ----------------------------------------------------
# Standalone use
# ----------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tracking metrics from hands.jsonl.")
    parser.add_argument("--json", default=None, help="Path to hands.jsonl.")
    parser.add_argument(
        "--total-frames", type=int, default=None,
        help="Number of frames in the video (for an exact detection rate).",
    )
    parser.add_argument(
        "--switch-window", type=int, default=30,
        help="Max frames between tracks to count an ID switch (default 30).",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print only; do not write metrics.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    calculator = MetricsCalculator(
        args.json,
        total_frames=args.total_frames,
        switch_window_frames=args.switch_window,
    )
    calculator.print_summary()
    if not args.no_save:
        print(f"Saved metrics to {calculator.save()}")