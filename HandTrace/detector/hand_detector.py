"""
hand_detector.py

MediaPipe Hand Landmarker wrapper.

Responsibilities:
- Load the MediaPipe Hand Landmarker model
- Detect hands in a frame
- Return structured detection results
- Release resources safely

Usage:
    with HandDetector(CONFIG.paths.model) as detector:
        for frame in reader.frames():
            detections = detector.detect(frame.image, frame.timestamp_ms)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import CONFIG, Detection


# ---------------------------------------------------------
# Data classes
# ---------------------------------------------------------

@dataclass(frozen=True)
class Landmark:
    """A landmark in normalized image coordinates (x, y in 0..1)."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandDetection:
    handedness: str                    # "Left" or "Right"
    score: float                       # handedness confidence, 0..1
    landmarks: tuple[Landmark, ...]    # 21 landmarks


# ---------------------------------------------------------
# Detector
# ---------------------------------------------------------

class HandDetector:
    """Wrapper around MediaPipe Hand Landmarker (VIDEO running mode)."""

    def __init__(
        self,
        model_path: Path | str,
        detection: Detection = CONFIG.detection,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Hand landmarker model not found: {self.model_path}")

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=detection.max_hands,
            min_hand_detection_confidence=detection.min_detection_confidence,
            min_hand_presence_confidence=detection.min_presence_confidence,
            min_tracking_confidence=detection.min_tracking_confidence,
        )

        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    # -----------------------------------------------------

    def detect(self, image: np.ndarray, timestamp_ms: int) -> list[HandDetection]:
        """
        Detect hands in one frame.

        Parameters
        ----------
        image : np.ndarray
            BGR OpenCV image (uint8, H x W x 3).
        timestamp_ms : int
            Frame timestamp. Must increase on every call (MediaPipe requirement).
        """
        if self._landmarker is None:
            raise RuntimeError("HandDetector is closed")

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR image of shape (H, W, 3)")

        if timestamp_ms <= self._last_timestamp_ms:
            raise ValueError(
                f"Timestamps must be strictly increasing "
                f"(got {timestamp_ms} after {self._last_timestamp_ms})"
            )
        self._last_timestamp_ms = timestamp_ms

        # OpenCV gives BGR but MediaPipe expects RGB. Skipping this
        # conversion silently hurts detection accuracy.
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections: list[HandDetection] = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            best = handedness[0]
            detections.append(
                HandDetection(
                    handedness=best.category_name,
                    score=best.score,
                    landmarks=tuple(Landmark(lm.x, lm.y, lm.z) for lm in landmarks),
                )
            )

        return detections

    # -----------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources (safe to call more than once)."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
