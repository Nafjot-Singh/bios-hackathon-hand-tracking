"""
video_reader.py

Reads a video frame-by-frame using OpenCV.

Responsibilities:
- Open the input video
- Read frames one at a time (with frame index and timestamp)
- Expose video properties (FPS, width, height, frame count)
- Release resources properly

Example:
    with VideoReader(path, fallback_fps=30) as reader:
        print(reader)
        for frame_idx, timestamp_ms, frame in reader.frames():
            ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class Frame:
    """A single decoded video frame."""

    index: int            # 0-based frame number
    timestamp_ms: int     # time since start of video, in milliseconds
    image: np.ndarray     # BGR image (H x W x 3)


class VideoReader:
    """A small, safe wrapper around OpenCV's VideoCapture."""

    def __init__(self, video_path: Path | str, fallback_fps: float = 30.0):
        self.video_path = Path(video_path)

        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Could not open video: {self.video_path}")

        # Video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Some files report 0 / NaN FPS; fall back to a sane default.
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and fps > 0 else float(fallback_fps)

        self._next_index = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self.width, self.height

    @property
    def duration_s(self) -> float | None:
        """Video length in seconds, or None if the frame count is unknown."""
        return self.frame_count / self.fps if self.frame_count > 0 else None

    @property
    def is_open(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Read the next frame.

        Returns:
            success (bool): True if a frame was read.
            frame (numpy.ndarray | None): BGR image, or None at end of video.
        """
        if not self.is_open:
            return False, None

        success, image = self.cap.read()
        if success:
            self._next_index += 1
        return success, image

    def frames(self) -> Iterator[Frame]:
        """
        Yield Frame objects with index and timestamp:

            for frame in reader.frames():
                frame.index, frame.timestamp_ms, frame.image
        """
        while True:
            index = self._next_index
            success, image = self.read()
            if not success:
                break

            # Timestamp from frame index keeps values strictly increasing,
            # which MediaPipe's VIDEO running mode requires.
            yield Frame(
                index=index,
                timestamp_ms=int(index * 1000 / self.fps),
                image=image,
            )

    def seek(self, frame_index: int) -> None:
        """Jump to a specific frame (0-based)."""
        if frame_index < 0 or (self.frame_count > 0 and frame_index >= self.frame_count):
            raise IndexError(f"Frame index out of range: {frame_index}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._next_index = frame_index

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def release(self) -> None:
        """Release video resources (safe to call more than once)."""
        if self.cap is not None:
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[np.ndarray]:
        """Allows `for image in reader:` (images only, no index)."""
        for frame in self.frames():
            yield frame.image

    def __len__(self) -> int:
        """Total number of frames (may be 0 if the container doesn't report it)."""
        return max(self.frame_count, 0)

    def __str__(self) -> str:
        frames = self.frame_count if self.frame_count > 0 else "unknown"
        return (
            f"VideoReader("
            f"path='{self.video_path.name}', "
            f"frames={frames}, "
            f"fps={self.fps:.2f}, "
            f"resolution={self.width}x{self.height})"
        )
