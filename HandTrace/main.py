"""
main.py

Runs hand detection and tracking on the input video, prints per-frame
results, writes the tracked hands to a JSON Lines file, (optionally) saves an
annotated debug video, detects objects with YOLO, works out what each
hand touches (contacts.csv, events_proposals.json, evidence frames), and
finally creates metrics, graphs and a report.

Usage:
    python main.py
    python main.py --max-frames 100     # quick test on the first 100 frames
    HT_DEBUG=0 python main.py           # progress + summary only
    HT_SAVE_DEBUG_VIDEO=0 python main.py  # skip the annotated video
    HT_OBJECTS=0 python main.py         # hands only: skip YOLO and the contact stage
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import CONFIG
from detector.hand_detector import HandDetector
from detector.landmark_utils import compute_centroid_px
from quality.metrics import MetricsCalculator
from tracker.smoother import LandmarkSmoother
from tracker.tracker import HandTracker
from utils.json_writer import JSONWriter, ObjectWriter
from utils.video_reader import VideoReader
from utils.visualization import Visualizer, VisualizerStyle

log = logging.getLogger("hand_tracking")

PROGRESS_EVERY = 30  # frames between progress lines when debug is off


@dataclass
class Stats:
    frames: int = 0
    frames_with_hands: int = 0
    total_hands: int = 0
    records_written: int = 0
    unique_hands: int = 0  # number of distinct track IDs created
    total_objects: int = 0  # object detections over all frames
    elapsed_s: float = 0.0
    video_fps: float = 0.0
    analyzer: Any = None  # ContactAnalyzer when the objects/contact stage ran

    @property
    def processing_fps(self) -> float:
        return self.frames / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def realtime_factor(self) -> float:
        """How many times faster than real time the video was processed."""
        return self.processing_fps / self.video_fps if self.video_fps > 0 else 0.0

    def summary(self) -> str:
        avg = self.total_hands / self.frames if self.frames else 0.0
        return (
            f"Done: {self.frames} frames in {self.elapsed_s:.1f}s\n"
            f"  Frames with hands : {self.frames_with_hands}\n"
            f"  Avg hands/frame   : {avg:.2f}\n"
            f"  Unique hands      : {self.unique_hands}\n"
            f"  Object detections : {self.total_objects}\n"
            f"  Records written   : {self.records_written}\n"
            f"  Video FPS         : {self.video_fps:.0f}\n"
            f"  Processing FPS    : {self.processing_fps:.0f}\n"
            f"  Speed             : {self.realtime_factor:.2f}x real-time"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hand detection on a video.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for quick tests).",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.DEBUG if CONFIG.debug else logging.INFO,
        format="%(message)s",
    )


def setup_objects(stack: ExitStack, video_stem: str, fps: float, stats: Stats):
    """
    Create the YOLO detector and the contact analyzer.

    Returns the detector, or None when the stage is off or could not start.
    A problem here never stops the hand pipeline; it is logged loudly instead.
    """
    if not CONFIG.objects.enabled:
        log.info("Objects and contact: off")
        return None

    try:
        from contact.analyzer import ContactAnalyzer
        from objects.object_detector import ObjectDetector

        detector = ObjectDetector()
        # Works whether or not ObjectDetector is a context manager.
        if hasattr(detector, "__enter__") and hasattr(detector, "__exit__"):
            detector = stack.enter_context(detector)

        stats.analyzer = ContactAnalyzer(video_stem, fps)
        log.info(
            "Objects and contact: on (%s)",
            getattr(CONFIG.objects, "weights", CONFIG.paths.yolo_model),
        )
        return detector
    except ImportError as exc:
        log.warning("Objects and contact stage skipped (import failed): %s", exc)
    except Exception:
        log.exception("Objects and contact stage skipped (could not start YOLO)")

    stats.analyzer = None
    return None


def process_video(max_frames: int | None) -> Stats:
    stats = Stats()
    start = time.perf_counter()

    with (
        VideoReader(CONFIG.paths.video, CONFIG.expected_fps) as reader,
        HandDetector(CONFIG.paths.model) as detector,
        JSONWriter(CONFIG.paths.output_json) as writer,
        ObjectWriter(CONFIG.paths.output_objects_json) as object_writer,
        ExitStack() as stack,  # holds the optional detector and visualizer
    ):
        log.info(reader)
        stats.video_fps = reader.fps

        # Frames we expect to process (video length, capped by --max-frames).
        total = len(reader) or None
        if max_frames is not None:
            total = min(total, max_frames) if total else max_frames

        # Objects and contact are optional (CONFIG.objects.enabled / HT_OBJECTS).
        object_detector = setup_objects(stack, CONFIG.paths.video.stem, reader.fps, stats)

        tracker = HandTracker()
        smoother = LandmarkSmoother(
            alpha=CONFIG.smoothing.alpha,
            max_missed=CONFIG.tracking.max_missing_frames,
        )

        # The debug video is optional (CONFIG.save_debug_video).
        visualizer: Visualizer | None = None
        if CONFIG.save_debug_video:
            visualizer = stack.enter_context(
                Visualizer(
                    reader.width,
                    reader.height,
                    reader.fps,
                    style=VisualizerStyle(color_mode="track_id"),
                )
            )

        try:
            for frame in reader.frames():
                if max_frames is not None and stats.frames >= max_frames:
                    break

                hands = detector.detect(frame.image, frame.timestamp_ms)

                tracks = tracker.update(
                    hands,
                    reader.width,
                    reader.height,
                )

                # Pass all active tracks so coasting tracks update their missed
                # counter and expired tracks get pruned from smoother state.
                smoother.smooth_tracks(tracks)

                visible = tracker.visible_tracks

                # Smoothing changed the landmarks, so refresh each centroid
                # (tracks are live objects). Missing hands are never filled in:
                # a hand that is not detected is simply not written (Hard Rule 2).
                for track in visible:
                    track.centroid = compute_centroid_px(
                        track.hand, reader.width, reader.height
                    )

                stats.records_written += writer.write_frame(
                    frame.index,
                    frame.timestamp_ms,
                    visible,
                )

                # Objects are detected on every frame with tracking on, so
                # the tracker keeps their IDs stable between frames.
                detections = (
                    object_detector.detect(frame.image, track=True)
                    if object_detector
                    else []
                )
                stats.total_objects += len(detections)

                if detections:
                    object_writer.write_frame(frame.index, frame.timestamp_ms, detections)

                if stats.analyzer is not None:
                    stats.analyzer.observe(
                        frame.index, visible, detections, reader.width, reader.height
                    )

                if visualizer is not None:
                    visualizer.write(frame.image, visible, frame.index, detections)

                # Count from the tracker so every output stays consistent.
                stats.frames += 1
                stats.total_hands += len(visible)
                stats.frames_with_hands += bool(visible)
                stats.unique_hands = max(
                    stats.unique_hands, max((t.track_id for t in tracks), default=0)
                )

                # Per-frame details (only when CONFIG.debug is on)
                log.debug(
                    "Frame %04d | Time %06d ms | Hands = %d | Tracks = %d | Objects = %d",
                    frame.index, frame.timestamp_ms, len(hands), len(tracks), len(detections),
                )
                for track in tracks:
                    state = (
                        "visible"
                        if track.updated_this_frame
                        else f"missing {track.missing_frames}"
                    )
                    log.debug(
                        "   ID %d: %s | Confidence = %.2f | %s",
                        track.track_id, track.hand.handedness, track.hand.score, state,
                    )

                # Lightweight progress when debug is off
                if not CONFIG.debug and stats.frames % PROGRESS_EVERY == 0:
                    if total:
                        log.info(
                            "Processed %d / %d (%d%%)",
                            stats.frames, total, 100 * stats.frames // total,
                        )
                    else:
                        log.info("Processed %d frames...", stats.frames)

        except KeyboardInterrupt:
            log.info("Interrupted by user; stopping early.")

    stats.elapsed_s = time.perf_counter() - start
    return stats


def run_contact_stage(stats: Stats) -> None:
    """
    Turn the collected observations into contacts.csv, events_proposals.json
    and evidence frames. A problem here is logged but never fails the run.
    """
    if stats.analyzer is None or stats.frames == 0:
        return
    try:
        from contact.output import save_all

        result = stats.analyzer.finalize()
        paths = save_all(result, CONFIG.paths.video, CONFIG.paths.video.stem)
        log.info(result.summary())
        for name, path in paths.items():
            log.info("%s saved to %s", name, path)
    except Exception:
        log.exception("Could not finish the objects and contact stage")


def run_event_naming(stats: Stats) -> None:
    """
    Validate candidate events and generate events_validated.json & actions.csv
    using Groq AI (openai/gpt-oss-120b). A problem here is logged but never fails the run.
    """
    if stats.frames == 0:
        return
    try:
        from event_namer import run_event_namer

        json_path, csv_path, calls = run_event_namer()
        log.info("Validated events saved to %s", json_path)
        log.info("Action timeline saved to %s (%d live Groq calls)", csv_path, calls)
    except Exception:
        log.exception("Could not finish the event naming stage")


def generate_outputs(stats: Stats) -> None:
    """
    Create metrics, graphs and the report from output/hands.jsonl.

    Each step is guarded separately: the video has already been processed
    and saved, so a problem here (for example matplotlib not installed)
    is logged but never turns a successful run into a failure.
    """
    if stats.frames == 0:
        log.info("No frames were processed; skipping metrics, graphs and report.")
        return

    out_dir: Path = CONFIG.paths.output_dir

    # 1. Metrics (the graphs and the report both need them)
    try:
        calculator = MetricsCalculator(
            CONFIG.paths.output_json,
            total_frames=stats.frames,
        )
        metrics = calculator.calculate()  # plain dict
        log.info("Metrics saved to %s", calculator.save(CONFIG.paths.output_metrics))
    except Exception:
        log.exception("Could not compute metrics; skipping graphs and report.")
        return

    # 2. Graphs (imported here so matplotlib is only needed for this step)
    graphs: dict[str, Path] = {}
    try:
        from quality.graph import GraphGenerator

        graphs = GraphGenerator(
            metrics,
            output_dir=out_dir / "graphs",
            jsonl_path=CONFIG.paths.output_json,
        ).generate_all()
        log.info("%d graphs saved to %s", len(graphs), out_dir / "graphs")
    except ImportError as exc:
        log.warning("Skipping graphs: %s (install with: pip install matplotlib)", exc)
    except Exception:
        log.exception("Could not generate graphs")

    # 3. Report (embeds whichever graphs were created)
    try:
        from quality.report import ReportGenerator

        report_path = ReportGenerator(
            metrics,
            graphs,
            output_path=out_dir / "report.html",
            source_name=CONFIG.paths.video.name,
        ).generate()
        log.info("Report saved to %s", report_path)
    except Exception:
        log.exception("Could not generate report")


def main() -> int:
    args = parse_args()
    setup_logging()

    try:
        CONFIG.ensure_dirs()      # create input/, outputs/, models/
        CONFIG.validate_inputs()  # check video and model exist
        stats = process_video(args.max_frames)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        # Expected problems (missing files, bad input): short message.
        log.error("Error: %s", exc)
        return 1
    except Exception:
        # Anything else (OpenCV, JSON, bugs): full traceback for debugging.
        log.exception("Unexpected error")
        return 1

    log.info(stats.summary())
    log.info("Results saved to %s", CONFIG.paths.output_json)
    if CONFIG.objects.enabled:
        log.info("Object detections saved to %s", CONFIG.paths.output_objects_json)
    if CONFIG.save_debug_video:
        log.info("Debug video saved to %s", CONFIG.paths.output_video)

    run_contact_stage(stats)
    run_event_naming(stats)
    generate_outputs(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())