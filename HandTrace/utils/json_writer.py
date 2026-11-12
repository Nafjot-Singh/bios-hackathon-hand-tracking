"""
json_writer.py

Writes tracked hand data to a JSON Lines (.jsonl) file.

Responsibilities
----------------
- Open the output JSONL file
- Write one JSON object per visible hand
- Flush data periodically so a crash loses little
- Close the file automatically

Each line in hands.jsonl represents ONE hand in ONE frame.

Example
-------
{"frame":0,"track_id":1,...}
{"frame":0,"track_id":2,...}
{"frame":1,"track_id":1,...}

Usage
-----
    with JSONWriter() as writer:
        for frame in reader.frames():
            ...
            writer.write_frame(frame.index, frame.timestamp_ms, tracker.visible_tracks)

ObjectWriter writes one JSON object per detected object per frame to objects.jsonl.

Example
-------
{"frame":0,"timestamp_ms":0,"label":"handheld tool","kind":"portable",...}

Usage
-----
    with ObjectWriter() as obj_writer:
        for frame in reader.frames():
            ...
            obj_writer.write_frame(frame.index, frame.timestamp_ms, detections)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Iterable

from config import CONFIG

if TYPE_CHECKING:  # type hints only
    from objects.object_detector import ObjectDetection
    from tracker.tracker import Track

# Decimal places kept in the output.
PIXEL_DECIMALS = 2       # centroid, in pixels
SCORE_DECIMALS = 4       # handedness confidence
LANDMARK_DECIMALS = 6    # normalized landmark coordinates


class JSONWriter:
    """Writes tracked hand data into a JSON Lines file."""

    def __init__(
        self,
        output_path: Path | str | None = None,
        flush_every_frames: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        output_path:
            Destination file. Defaults to CONFIG.paths.output_json.
            Existing files are overwritten.
        flush_every_frames:
            Flush to disk after this many written frames (0 = only on
            close). Keeps most of the data if the program crashes.
        """
        if flush_every_frames < 0:
            raise ValueError("flush_every_frames must be >= 0")

        self.output_path = Path(output_path or CONFIG.paths.output_json)
        self.flush_every_frames = flush_every_frames

        self.frames_written = 0
        self.records_written = 0

        # Set before opening so close()/__del__ are safe if open() fails.
        self._file: IO[str] | None = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")

    # -----------------------------------------------------
    # Writing
    # -----------------------------------------------------

    def write_frame(
        self,
        frame_index: int,
        timestamp_ms: int,
        tracks: Iterable[Track],
    ) -> int:
        """
        Write the visible tracks of one frame.

        Tracks that are only being kept alive (not detected in this frame)
        are skipped, so passing `tracker.active_tracks` or
        `tracker.visible_tracks` gives the same output.

        Returns
        -------
        int
            Number of records (hands) written for this frame.
        """
        file = self._require_open()

        lines = [
            json.dumps(
                self._to_record(frame_index, timestamp_ms, track),
                separators=(",", ":"),  # compact: no spaces
                allow_nan=False,        # never write invalid JSON (NaN/Infinity)
            )
            for track in tracks
            if track.updated_this_frame
        ]

        if lines:
            file.write("\n".join(lines) + "\n")

        self.records_written += len(lines)
        self.frames_written += 1

        if self.flush_every_frames and self.frames_written % self.flush_every_frames == 0:
            file.flush()

        return len(lines)

    @staticmethod
    def _to_record(frame_index: int, timestamp_ms: int, track: Track) -> dict[str, Any]:
        """Convert one track into a JSON-serializable dictionary."""
        hand = track.hand
        return {
            "frame": frame_index,
            "timestamp_ms": timestamp_ms,
            "track_id": track.track_id,
            "handedness": hand.handedness,
            "confidence": round(hand.score, SCORE_DECIMALS),
            "centroid": {
                "x": round(track.centroid[0], PIXEL_DECIMALS),
                "y": round(track.centroid[1], PIXEL_DECIMALS),
            },
            "landmarks": [
                {
                    "x": round(lm.x, LANDMARK_DECIMALS),
                    "y": round(lm.y, LANDMARK_DECIMALS),
                    "z": round(lm.z, LANDMARK_DECIMALS),
                }
                for lm in hand.landmarks
            ],
        }

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._file is None or self._file.closed

    def flush(self) -> None:
        """Force buffered data to disk."""
        self._require_open().flush()

    def close(self) -> None:
        """Flush and close the file (safe to call more than once)."""
        if self._file is not None and not self._file.closed:
            self._file.close()  # close() flushes remaining data

    def _require_open(self) -> IO[str]:
        if self._file is None or self._file.closed:
            raise ValueError("JSONWriter is closed")
        return self._file

    def __enter__(self) -> JSONWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ObjectWriter:
    """
    Writes one JSON record per detected object per frame to objects.jsonl.

    Frames with zero detections are skipped entirely.
    Each line represents ONE object detection in ONE frame.

    Example line
    ------------
    {"frame":42,"timestamp_ms":1400,"label":"handheld tool","kind":"portable",
     "confidence":0.81,"bbox":[120,55,310,290],"track_id":3}
    """

    def __init__(
        self,
        output_path: Path | str | None = None,
        flush_every_frames: int = 30,
    ) -> None:
        """
        Parameters
        ----------
        output_path:
            Destination file. Defaults to CONFIG.paths.output_objects_json.
            Existing files are overwritten.
        flush_every_frames:
            Flush to disk after this many written frames (0 = only on close).
        """
        if flush_every_frames < 0:
            raise ValueError("flush_every_frames must be >= 0")

        self.output_path = Path(output_path or CONFIG.paths.output_objects_json)
        self.flush_every_frames = flush_every_frames

        self.frames_written = 0   # frames that had at least one detection
        self.records_written = 0  # total detection records written

        self._file: IO[str] | None = None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", encoding="utf-8")

    # -----------------------------------------------------
    # Writing
    # -----------------------------------------------------

    def write_frame(
        self,
        frame_index: int,
        timestamp_ms: int,
        detections: Iterable[ObjectDetection],
    ) -> int:
        """
        Write all detections for one frame.

        Frames with no detections are silently skipped.

        Returns
        -------
        int
            Number of records written for this frame.
        """
        file = self._require_open()

        lines = [
            json.dumps(
                self._to_record(frame_index, timestamp_ms, det),
                separators=(",", ":"),
                allow_nan=False,
            )
            for det in detections
        ]

        if not lines:
            return 0

        file.write("\n".join(lines) + "\n")
        self.records_written += len(lines)
        self.frames_written += 1

        if self.flush_every_frames and self.frames_written % self.flush_every_frames == 0:
            file.flush()

        return len(lines)

    @staticmethod
    def _to_record(
        frame_index: int,
        timestamp_ms: int,
        det: ObjectDetection,
    ) -> dict[str, Any]:
        """Convert one ObjectDetection into a JSON-serializable dict."""
        x1, y1, x2, y2 = det.bbox
        return {
            "frame": frame_index,
            "timestamp_ms": timestamp_ms,
            "label": det.label,
            "kind": det.kind,
            "confidence": round(det.confidence, SCORE_DECIMALS),
            "bbox": [x1, y1, x2, y2],
            "track_id": det.track_id,
        }

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._file is None or self._file.closed

    def flush(self) -> None:
        """Force buffered data to disk."""
        self._require_open().flush()

    def close(self) -> None:
        """Flush and close the file (safe to call more than once)."""
        if self._file is not None and not self._file.closed:
            self._file.close()

    def _require_open(self) -> IO[str]:
        if self._file is None or self._file.closed:
            raise ValueError("ObjectWriter is closed")
        return self._file

    def __enter__(self) -> ObjectWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
