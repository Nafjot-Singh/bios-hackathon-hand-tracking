"""
visualization.py

Draws tracked hands onto video frames and saves the annotated video.

Responsibilities:
- Open an OpenCV VideoWriter (mp4v codec)
- Draw skeleton, landmarks, bounding box, centroid and label for each
  visible Track (optionally color-coded by track ID or handedness)
- Optionally overlay the frame number and processing FPS
- Write annotated frames to the output video
- Release the VideoWriter safely

This module DOES NOT:
- Detect hands
- Assign track IDs
- Smooth landmarks
- Write JSON

Usage:
    with Visualizer(reader.width, reader.height, reader.fps) as visualizer:
        for frame in reader.frames():
            ...
            visualizer.write(frame.image, tracker.visible_tracks, frame.index)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import time

import cv2
import numpy as np

from config import CONFIG
from detector.landmark_utils import compute_bbox_px, compute_landmarks_px

if TYPE_CHECKING:  # type hints only
    from objects.object_detector import ObjectDetection
    from tracker.tracker import Track

# Colors are BGR (OpenCV order).
Color = tuple[int, int, int]
Point = tuple[int, int]

GREEN: Color = (0, 255, 0)
YELLOW: Color = (0, 255, 255)
BLUE: Color = (255, 0, 0)
RED: Color = (0, 0, 255)
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
ORANGE: Color = (0, 165, 255)
MAGENTA: Color = (255, 0, 255)
CYAN: Color = (255, 255, 0)

# "fixed":      every hand uses the style's landmark/bbox colors
# "track_id":   each track ID gets its own color from `id_palette`
# "handedness": Left and Right hands get different colors
ColorMode = Literal["fixed", "track_id", "handedness"]

# Standard MediaPipe hand skeleton, used only if the installed MediaPipe
# version does not expose HAND_CONNECTIONS.
_FALLBACK_HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),    # pinky
    (0, 17),                                   # palm edge
)


def _load_hand_connections() -> tuple[tuple[int, int], ...]:
    """
    Return MediaPipe's HAND_CONNECTIONS as (start, end) landmark index pairs.

    Tries the Tasks API first, then the legacy `solutions` API, and finally
    falls back to the standard hand skeleton.
    """
    try:
        from mediapipe.tasks.python.vision import HandLandmarksConnections

        return tuple(
            (int(c.start), int(c.end))
            for c in HandLandmarksConnections.HAND_CONNECTIONS
        )
    except (ImportError, AttributeError):
        pass

    try:
        import mediapipe as mp

        return tuple(
            sorted((int(a), int(b)) for a, b in mp.solutions.hands.HAND_CONNECTIONS)
        )
    except (ImportError, AttributeError):
        return _FALLBACK_HAND_CONNECTIONS


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = _load_hand_connections()


@dataclass(frozen=True)
class VisualizerStyle:
    """Colors and sizes used when drawing."""

    landmark_color: Color = GREEN
    connection_color: Color = YELLOW
    bbox_color: Color = BLUE
    centroid_color: Color = RED
    text_color: Color = WHITE
    portable_object_color: Color = ORANGE   # detected portable objects
    fixed_surface_color: Color = (160, 160, 160)  # detected fixed surfaces
    text_outline_color: Color = BLACK  # thin outline keeps white text readable

    # Color coding for landmarks and bounding boxes (connections stay yellow
    # and the centroid stays red in every mode).
    color_mode: ColorMode = "fixed"
    id_palette: tuple[Color, ...] = (GREEN, BLUE, RED, ORANGE, MAGENTA, CYAN)
    left_color: Color = BLUE
    right_color: Color = GREEN

    # Text overlay in the top-left corner.
    show_frame_number: bool = True   # needs frame_index passed to write()
    show_fps: bool = False           # measured processing FPS

    landmark_radius: int = 3
    connection_thickness: int = 2
    bbox_thickness: int = 2
    centroid_radius: int = 5

    bbox_padding: int = 10             # extra pixels around the landmarks
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.55
    text_thickness: int = 1
    line_spacing: int = 6              # pixels between label lines

    def __post_init__(self) -> None:
        if self.color_mode not in ("fixed", "track_id", "handedness"):
            raise ValueError(f"Unknown color_mode: {self.color_mode!r}")
        if not self.id_palette:
            raise ValueError("id_palette must not be empty")


class Visualizer:
    """Draws Track objects on frames and writes them to a video file."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: Path | str | None = None,
        style: VisualizerStyle | None = None,
    ) -> None:
        """
        Parameters
        ----------
        width, height:
            Frame size in pixels. Every frame passed to `write()` must match.
        fps:
            Frame rate of the output video.
        output_path:
            Destination file. Defaults to CONFIG.paths.output_video.
        style:
            Colors and sizes. Defaults to VisualizerStyle().

        Raises
        ------
        ValueError
            If width, height or fps is not positive.
        RuntimeError
            If the video file cannot be opened for writing.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid frame size: {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0 (got {fps})")

        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = Path(output_path or CONFIG.paths.output_video)
        self.style = style or VisualizerStyle()
        self.frames_written = 0

        # Processing-FPS measurement (smoothed over recent write() calls).
        self._last_write_time: float | None = None
        self._processing_fps: float | None = None

        # Set before opening so close()/__del__ are safe if opening fails.
        self._writer: cv2.VideoWriter | None = None

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Could not open video writer: {self.output_path}")

        self._writer = writer

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------

    def write(
        self,
        frame: np.ndarray,
        tracks: Iterable[Track],
        frame_index: int | None = None,
        objects: Iterable[ObjectDetection] | None = None,
    ) -> None:
        """
        Draw tracks (and optionally detected objects) on a copy of `frame`
        and write it to the video.

        The input frame is not modified.

        Parameters
        ----------
        frame:
            BGR image (uint8, H x W x 3) matching the video size.
        tracks:
            Tracks to draw; only those with `updated_this_frame` are shown.
        frame_index:
            Frame number for the "Frame: N" overlay. Omit to hide it.
        objects:
            Detected objects to draw (boxes with label and ID), if any.

        Raises
        ------
        ValueError
            If the frame is not uint8 or does not match the video size.
        RuntimeError
            If the visualizer has been closed.
        """
        if self._writer is None:
            raise RuntimeError("Visualizer is closed")

        if frame.dtype != np.uint8:
            raise ValueError(f"Frame must be uint8 (got {frame.dtype})")

        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Frame shape {frame.shape} does not match "
                f"expected ({self.height}, {self.width}, 3)"
            )

        self._update_processing_fps()

        annotated = frame.copy()
        if objects:
            self._draw_objects(annotated, objects)
        self.draw_tracks(annotated, tracks)
        self._draw_overlay(annotated, frame_index)
        self._writer.write(annotated)
        self.frames_written += 1

    def draw_tracks(self, frame: np.ndarray, tracks: Iterable[Track]) -> None:
        """
        Draw all visible tracks onto `frame` in place.

        Tracks that are only being kept alive (no detection in this frame)
        are skipped.
        """
        for track in tracks:
            if not track.updated_this_frame:
                continue

            points = [
                (round(x), round(y))
                for x, y in compute_landmarks_px(track.hand, self.width, self.height)
            ]

            # Skeleton first so landmark dots sit on top of the lines.
            landmark_color, bbox_color = self._track_colors(track)

            self._draw_connections(frame, points)
            self._draw_landmarks(frame, points, landmark_color)

            bbox = self._draw_bbox(frame, track, bbox_color)
            self._draw_centroid(frame, track)
            self._draw_label(frame, track, bbox)

    def close(self) -> None:
        """Finish and release the video file (safe to call more than once)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # -----------------------------------------------------
    # Drawing helpers
    # -----------------------------------------------------

    def _draw_objects(self, frame: np.ndarray, objects: Iterable[ObjectDetection]) -> None:
        """Draw object boxes with "label #id confidence"."""
        style = self.style
        for obj in objects:
            color = style.portable_object_color if obj.kind == "portable" else style.fixed_surface_color
            x0, y0, x1, y1 = (round(v) for v in obj.bbox)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, style.bbox_thickness)
            ident = f" #{obj.object_id}" if obj.object_id is not None else ""
            self._put_text(frame, f"{obj.label}{ident} {obj.confidence:.2f}", (x0 + 2, max(14, y0 - 4)))

    def _track_colors(self, track: Track) -> tuple[Color, Color]:
        """
        Pick (landmark_color, bbox_color) for a track according to
        `style.color_mode`.
        """
        style = self.style

        if style.color_mode == "track_id":
            color = style.id_palette[(track.track_id - 1) % len(style.id_palette)]
            return color, color

        if style.color_mode == "handedness":
            label = track.hand.handedness.lower()
            if label == "left":
                return style.left_color, style.left_color
            if label == "right":
                return style.right_color, style.right_color

        return style.landmark_color, style.bbox_color

    def _draw_landmarks(
        self, frame: np.ndarray, points: list[Point], color: Color
    ) -> None:
        """Draw the 21 landmarks as filled dots."""
        for point in points:
            cv2.circle(
                frame, point, self.style.landmark_radius,
                color, thickness=-1, lineType=cv2.LINE_AA,
            )

    def _draw_connections(self, frame: np.ndarray, points: list[Point]) -> None:
        """Draw the hand skeleton as yellow lines between landmarks."""
        for start, end in HAND_CONNECTIONS:
            if start >= len(points) or end >= len(points):
                continue
            cv2.line(
                frame, points[start], points[end],
                self.style.connection_color, self.style.connection_thickness,
                lineType=cv2.LINE_AA,
            )

    def _draw_bbox(
        self, frame: np.ndarray, track: Track, color: Color
    ) -> tuple[int, int, int, int]:
        """
        Draw the bounding box around the hand.

        Returns
        -------
        tuple[int, int, int, int]
            The drawn box as (x_min, y_min, x_max, y_max), padded and
            clamped to the frame.
        """
        x0, y0, x1, y1 = compute_bbox_px(track.hand, self.width, self.height)
        pad = self.style.bbox_padding

        box = (
            max(0, round(x0) - pad),
            max(0, round(y0) - pad),
            min(self.width - 1, round(x1) + pad),
            min(self.height - 1, round(y1) + pad),
        )
        cv2.rectangle(
            frame, box[:2], box[2:], color, self.style.bbox_thickness,
        )
        return box

    def _draw_centroid(self, frame: np.ndarray, track: Track) -> None:
        """Draw the red centroid dot (track.centroid is in pixels)."""
        center = (round(track.centroid[0]), round(track.centroid[1]))
        cv2.circle(
            frame, center, self.style.centroid_radius,
            self.style.centroid_color, thickness=-1, lineType=cv2.LINE_AA,
        )

    def _draw_label(
        self,
        frame: np.ndarray,
        track: Track,
        bbox: tuple[int, int, int, int],
    ) -> None:
        """
        Draw the ID, handedness and confidence above the bounding box.

        If there is no room above the box (hand near the top edge), the
        label is drawn just below it instead.
        """
        lines = [
            f"ID: {track.track_id}",
            f"{track.hand.handedness} Hand",
            f"Conf: {track.hand.score:.2f}",
        ]

        style = self.style
        sizes = [
            cv2.getTextSize(text, style.font, style.font_scale, style.text_thickness)[0]
            for text in lines
        ]
        line_height = max(h for _, h in sizes) + style.line_spacing
        block_height = line_height * len(lines)

        x0, y0, _, y1 = bbox
        x = min(max(0, x0), max(0, self.width - max(w for w, _ in sizes) - 1))

        if y0 - block_height >= 0:
            top = y0 - block_height - 2          # above the box
        else:
            top = min(y1 + 4, self.height - block_height)  # below the box

        for i, text in enumerate(lines):
            baseline_y = int(top + (i + 1) * line_height - style.line_spacing)
            baseline_y = max(sizes[i][1], baseline_y)

            self._put_text(frame, text, (x, baseline_y))

    def _draw_overlay(self, frame: np.ndarray, frame_index: int | None) -> None:
        """Draw "Frame: N" and "FPS: x" in the top-left corner."""
        style = self.style
        lines: list[str] = []

        if style.show_frame_number and frame_index is not None:
            lines.append(f"Frame: {frame_index}")
        if style.show_fps and self._processing_fps is not None:
            lines.append(f"FPS: {self._processing_fps:.1f}")

        y = 0
        for text in lines:
            (_, text_h), _ = cv2.getTextSize(
                text, style.font, style.font_scale, style.text_thickness
            )
            y += text_h + style.line_spacing
            self._put_text(frame, text, (8, y + 4))

    def _put_text(self, frame: np.ndarray, text: str, origin: Point) -> None:
        """Draw white text with a dark outline so it is readable anywhere."""
        style = self.style
        cv2.putText(
            frame, text, origin, style.font, style.font_scale,
            style.text_outline_color, style.text_thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, origin, style.font, style.font_scale,
            style.text_color, style.text_thickness, cv2.LINE_AA,
        )

    def _update_processing_fps(self) -> None:
        """Track how fast frames arrive (exponential moving average)."""
        now = time.perf_counter()
        if self._last_write_time is not None:
            elapsed = now - self._last_write_time
            if elapsed > 0:
                instant = 1.0 / elapsed
                self._processing_fps = (
                    instant
                    if self._processing_fps is None
                    else 0.9 * self._processing_fps + 0.1 * instant
                )
        self._last_write_time = now

    # -----------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------

    def __enter__(self) -> Visualizer:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
