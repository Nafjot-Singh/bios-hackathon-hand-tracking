"""
graph.py

Draws tracking graphs (PNG files) from the metrics computed by metrics.py.

Input:
    - Metrics dictionary from MetricsCalculator.calculate() (or metrics.json)
    - Optionally hands.jsonl, for graphs that need per-frame data
      (hands over time, confidence distribution)

Output:
    PNG files in output/graphs/

This module DOES NOT:
- Compute metrics
- Generate reports
- Modify tracking results

Those responsibilities belong to metrics.py and report.py.

Graphs
------
    hands_over_time     hands detected in every frame        (needs hands.jsonl)
    hands_per_frame     how many frames had 0, 1, 2 ... hands
    confidence          handedness confidence distribution   (needs hands.jsonl)
    handedness          Left vs Right records
    track_timeline      when each track was alive, with ID switches
    track_lifetimes     visible frames and gap frames per track

Usage:
    metrics = MetricsCalculator(total_frames=1200).calculate()
    graphs = GraphGenerator(metrics).generate_all()   # {"name": Path, ...}
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed

import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from config import CONFIG  # noqa: E402

log = logging.getLogger(__name__)

# Colors
LEFT_COLOR = "#2563eb"
RIGHT_COLOR = "#16a34a"
OTHER_COLOR = "#6b7280"
GAP_COLOR = "#f59e0b"
SWITCH_COLOR = "#dc2626"
NEUTRAL_COLOR = "#4b5563"


def _handedness_color(label: str) -> str:
    """Blue for Left, green for Right, gray for anything else."""
    return {"left": LEFT_COLOR, "right": RIGHT_COLOR}.get(label.lower(), OTHER_COLOR)


def _load_series(path: Path) -> tuple[dict[int, int], list[float]]:
    """
    Stream hands.jsonl and return (hands per frame, all confidences).

    Unreadable lines are skipped; metrics.py already reports them.
    """
    hands_per_frame: dict[int, int] = {}
    confidences: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                frame = int(record["frame"])
                confidence = float(record["confidence"])
            except (ValueError, KeyError, TypeError):
                continue
            hands_per_frame[frame] = hands_per_frame.get(frame, 0) + 1
            confidences.append(confidence)

    return hands_per_frame, confidences


class GraphGenerator:
    """Creates PNG graphs from a metrics dictionary."""

    def __init__(
        self,
        metrics: Mapping[str, Any],
        output_dir: Path | str | None = None,
        jsonl_path: Path | str | None = None,
        dpi: int = 150,
        max_tracks: int = 40,
    ) -> None:
        """
        Parameters
        ----------
        metrics:
            Output of MetricsCalculator.calculate() or a loaded metrics.json.
        output_dir:
            Where PNGs are saved. Defaults to <output dir>/graphs.
        jsonl_path:
            hands.jsonl, used for per-frame graphs. Defaults to
            CONFIG.paths.output_json. If the file is missing, those graphs
            are skipped.
        dpi:
            Image resolution.
        max_tracks:
            Track graphs show at most this many tracks (in ID order) so they
            stay readable.
        """
        if dpi <= 0 or max_tracks <= 0:
            raise ValueError("dpi and max_tracks must be > 0")

        self.metrics: dict[str, Any] = dict(metrics)
        self.output_dir = Path(output_dir or CONFIG.paths.output_dir / "graphs")
        self.jsonl_path = Path(jsonl_path or CONFIG.paths.output_json)
        self.dpi = dpi
        self.max_tracks = max_tracks

        self._series: tuple[dict[int, int], list[float]] | None = None

    @classmethod
    def from_file(
        cls, metrics_path: Path | str | None = None, **kwargs: Any
    ) -> GraphGenerator:
        """Create a generator from a saved metrics.json."""
        path = Path(metrics_path or CONFIG.paths.output_metrics)
        if not path.is_file():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        return cls(json.loads(path.read_text(encoding="utf-8")), **kwargs)

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------

    def generate_all(self) -> dict[str, Path]:
        """
        Draw every graph that has data.

        Returns
        -------
        dict[str, Path]
            Graph name -> PNG path, in display order. Graphs without data
            are left out.
        """
        plots = {
            "hands_over_time": self.plot_hands_over_time,
            "hands_per_frame": self.plot_hands_per_frame,
            "confidence": self.plot_confidence,
            "handedness": self.plot_handedness,
            "track_timeline": self.plot_track_timeline,
            "track_lifetimes": self.plot_track_lifetimes,
        }

        results: dict[str, Path] = {}
        for name, plot in plots.items():
            path = plot()
            if path is not None:
                results[name] = path
        return results

    # ----------------------------------------------------
    # Graphs that only need metrics
    # ----------------------------------------------------

    def plot_hands_per_frame(self) -> Path | None:
        """Bar chart: number of frames that had 0, 1, 2 ... hands."""
        counts = {int(k): int(v) for k, v in self.metrics.get("hands_per_frame_histogram", {}).items()}
        without = int(self.metrics.get("frames_without_hands", 0))
        if without:
            counts[0] = without
        if not counts:
            return None

        labels = sorted(counts)
        values = [counts[k] for k in labels]

        fig, ax = plt.subplots(figsize=(7, 4))
        positions = list(range(len(labels)))  # numeric positions: avoids
        bars = ax.bar(positions, values, color=NEUTRAL_COLOR)  # categorical-unit warnings
        ax.set_xticks(positions)
        ax.set_xticklabels([str(k) for k in labels])
        ax.bar_label(bars, padding=2, fontsize=9)
        ax.set_xlabel("Hands in frame")
        ax.set_ylabel("Number of frames")
        ax.set_title("Frames by number of hands")
        ax.margins(y=0.12)
        return self._save(fig, "hands_per_frame")

    def plot_handedness(self) -> Path | None:
        """Bar chart: Left vs Right records."""
        counts: dict[str, int] = dict(self.metrics.get("handedness_counts", {}))
        if not counts:
            return None

        total = sum(counts.values())
        labels = list(counts)
        values = [counts[k] for k in labels]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, values, color=[_handedness_color(k) for k in labels])
        ax.bar_label(
            bars,
            labels=[f"{v} ({v / total:.0%})" for v in values],
            padding=2,
            fontsize=9,
        )
        ax.set_ylabel("Records")
        ax.set_title("Handedness")
        ax.margins(y=0.15)
        return self._save(fig, "handedness")

    def plot_track_timeline(self) -> Path | None:
        """
        Timeline: one bar per track from its first to last frame.

        Dashed red arrows show estimated ID switches (old track -> new track).
        """
        tracks, truncated = self._tracks()
        if not tracks:
            return None

        rows = {t["track_id"]: i for i, t in enumerate(tracks)}
        total_frames = max(int(self.metrics.get("total_frames", 0)), 1)

        fig, ax = plt.subplots(figsize=(10, max(3.0, 0.4 * len(tracks) + 1.8)))

        for track in tracks:
            y = rows[track["track_id"]]
            ax.barh(
                y, track["span"], left=track["first_frame"], height=0.6,
                color=_handedness_color(track["handedness"]), edgecolor="white",
            )
            if track["gap_frames"]:
                ax.text(
                    track["last_frame"] + total_frames * 0.01, y,
                    f"{track['gap_frames']} gap frames",
                    va="center", fontsize=8, color=GAP_COLOR,
                )

        for event in self.metrics.get("id_switch_events", []):
            old, new = event["from_track"], event["to_track"]
            if old in rows and new in rows:
                ax.annotate(
                    "",
                    xy=(event["first_frame_new"], rows[new]),
                    xytext=(event["last_frame_old"], rows[old]),
                    arrowprops={"arrowstyle": "->", "color": SWITCH_COLOR,
                                "lw": 1.5, "linestyle": "--"},
                )

        ax.set_yticks(range(len(tracks)))
        ax.set_yticklabels([f"ID {t['track_id']}" for t in tracks])
        ax.invert_yaxis()
        ax.set_xlim(0, total_frames)
        ax.set_xlabel("Frame")
        ax.set_title(self._truncated_title("Track timeline", truncated))
        ax.grid(axis="x", alpha=0.3)

        handles: list[Any] = [
            Patch(color=_handedness_color(label), label=label)
            for label in sorted({t["handedness"] for t in tracks})
        ]
        if self.metrics.get("id_switch_events"):
            handles.append(
                Line2D([0], [0], color=SWITCH_COLOR, lw=1.5, linestyle="--",
                       label="ID switch")
            )
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        return self._save(fig, "track_timeline")

    def plot_track_lifetimes(self) -> Path | None:
        """Bars: frames each track was visible, plus frames it was missing."""
        tracks, truncated = self._tracks()
        if not tracks:
            return None

        ids = [f"ID {t['track_id']}" for t in tracks]
        visible = [t["visible_frames"] for t in tracks]
        gaps = [t["gap_frames"] for t in tracks]

        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(tracks) + 1.8)))
        bars = ax.barh(
            ids, visible,
            color=[_handedness_color(t["handedness"]) for t in tracks],
        )
        ax.barh(
            ids, gaps, left=visible, color=GAP_COLOR, hatch="//",
            edgecolor="white", label="Missing (gap) frames",
        )
        ax.bar_label(bars, padding=3, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Frames")
        ax.set_title(self._truncated_title("Track lifetimes", truncated))
        ax.grid(axis="x", alpha=0.3)
        ax.margins(x=0.1)

        handles: list[Any] = [
            Patch(color=_handedness_color(label), label=f"{label} (visible)")
            for label in sorted({t["handedness"] for t in tracks})
        ]
        if any(gaps):
            handles.append(Patch(facecolor=GAP_COLOR, hatch="//", edgecolor="white",
                                 label="Missing (gap) frames"))
        ax.legend(handles=handles, loc="lower right", fontsize=8)
        return self._save(fig, "track_lifetimes")

    # ----------------------------------------------------
    # Graphs that need hands.jsonl
    # ----------------------------------------------------

    def plot_hands_over_time(self) -> Path | None:
        """Step plot: hands detected in every frame of the video."""
        series = self._load_raw()
        if series is None or not series[0]:
            return None
        hands_per_frame, _ = series

        length = max(int(self.metrics.get("total_frames", 0)), max(hands_per_frame) + 1)
        counts = np.zeros(length, dtype=int)  # frames without records stay 0
        for frame, n in hands_per_frame.items():
            counts[frame] = n

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.fill_between(np.arange(length), counts, step="mid", color=NEUTRAL_COLOR, alpha=0.35)
        ax.step(np.arange(length), counts, where="mid", color=NEUTRAL_COLOR, lw=1)
        ax.set_xlim(0, max(length - 1, 1))
        ax.set_ylim(0, max(int(counts.max()), 1) + 0.5)
        ax.set_yticks(range(0, int(counts.max()) + 1))
        ax.set_xlabel("Frame")
        ax.set_ylabel("Hands")
        ax.set_title("Hands detected per frame")
        ax.grid(axis="y", alpha=0.3)
        return self._save(fig, "hands_over_time")

    def plot_confidence(self) -> Path | None:
        """Histogram of handedness confidence with mean and threshold lines."""
        series = self._load_raw()
        if series is None or not series[1]:
            return None
        confidences = series[1]

        lowest = math.floor(min(confidences) * 20) / 20
        lowest = min(lowest, 0.95)  # keep a visible range when all values are ~1
        bins = np.linspace(lowest, 1.0, 21)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(confidences, bins=bins, color=NEUTRAL_COLOR, edgecolor="white")
        ax.axvline(
            float(np.mean(confidences)), color=RIGHT_COLOR, lw=1.5,
            label=f"Mean {np.mean(confidences):.3f}",
        )
        threshold = float(self.metrics.get("low_confidence_threshold", 0.0))
        if threshold > 0:
            ax.axvline(
                threshold, color=SWITCH_COLOR, lw=1.5, linestyle="--",
                label=f"Low-confidence threshold {threshold:g}",
            )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Records")
        ax.set_title("Confidence distribution")
        ax.legend(fontsize=8)
        return self._save(fig, "confidence")

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------

    def _tracks(self) -> tuple[list[dict[str, Any]], bool]:
        """Return (tracks to draw, whether the list was cut short)."""
        tracks = sorted(self.metrics.get("tracks", []), key=lambda t: t["track_id"])
        return tracks[: self.max_tracks], len(tracks) > self.max_tracks

    def _truncated_title(self, title: str, truncated: bool) -> str:
        return f"{title} (first {self.max_tracks} tracks)" if truncated else title

    def _load_raw(self) -> tuple[dict[int, int], list[float]] | None:
        """Read hands.jsonl once (cached). None if the file does not exist."""
        if self._series is None:
            if not self.jsonl_path.is_file():
                log.warning("%s not found; skipping per-frame graphs", self.jsonl_path)
                return None
            self._series = _load_series(self.jsonl_path)
        return self._series

    def _save(self, fig: Figure, name: str) -> Path:
        """Tidy, save and close a figure."""
        for ax in fig.axes:
            ax.spines[["top", "right"]].set_visible(False)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, facecolor="white")
        plt.close(fig)  # free memory
        return path


if __name__ == "__main__":
    # python -m quality.graph  -> draws graphs from output/metrics.json
    for graph_name, graph_path in GraphGenerator.from_file().generate_all().items():
        print(f"{graph_name:<16} {graph_path}")
