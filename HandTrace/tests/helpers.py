"""Test helpers: stand-ins for detector output (no MediaPipe needed)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandDetection:
    handedness: str
    score: float
    landmarks: tuple[Landmark, ...]


def make_hand(x: float, y: float = 0.5, side: str = "Left", score: float = 0.9) -> HandDetection:
    """A 21-landmark hand whose landmarks are spread slightly around (x, y)."""
    landmarks = tuple(Landmark(x + 0.002 * i, y + 0.002 * i, 0.0) for i in range(21))
    return HandDetection(side, score, landmarks)
