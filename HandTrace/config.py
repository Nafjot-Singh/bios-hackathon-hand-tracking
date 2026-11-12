"""
config.py

Global configuration and paths for HandTrace.

Holds settings for:
- File and directory paths (Paths)
- MediaPipe hand detection (Detection)
- Track association and persistence (Tracking)
- Landmark smoothing (Smoothing)
- YOLO-World object detection and ByteTrack tracking (Objects)
- Multi-cue contact estimation and state machine smoothing (Contact)
- Run-level options (Config, CONFIG)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _resolve_path(env_var: str, default: Path) -> Path:
    val = os.getenv(env_var)
    if not val:
        return default
    p = Path(val)
    if p.is_file() or p.is_dir():
        return p
    candidate = default.parent / val
    if candidate.exists():
        return candidate
    return p


# -------------------------------------------------
# Paths
# -------------------------------------------------

@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parent

    input_dir: Path = root / "input"
    output_dir: Path = root / "outputs"
    models_dir: Path = root / "models"

    # Input video & models
    video: Path = Path(os.getenv("HT_VIDEO", root / "input" / "demo.mp4"))
    model: Path = Path(os.getenv("HT_MODEL", root / "models" / "hand_landmarker.task"))
    yolo_model: Path = Path(
        os.getenv("HT_YOLO_WEIGHTS", root / "models" / "yolov8s-worldv2.pt")
    )

    # Outputs
    output_json: Path = output_dir / "hands.jsonl"
    output_objects_json: Path = output_dir / "objects.jsonl"
    output_video: Path = output_dir / "debug_video.mp4"
    output_metrics: Path = output_dir / "metrics.json"

    # Contact stage outputs
    contacts_csv: Path = output_dir / "contacts.csv"
    events_json: Path = output_dir / "events_proposals.json"
    evidence_dir: Path = output_dir / "evidence_frames"
    events_validated_json: Path = output_dir / "events_validated.json"
    actions_csv: Path = output_dir / "actions.csv"

    # Graphs & Report
    graphs_dir: Path = output_dir / "graphs"
    report_file: Path = output_dir / "report.html"


# -------------------------------------------------
# Detection (MediaPipe Hand Landmarker)
# -------------------------------------------------

@dataclass(frozen=True)
class Detection:
    max_hands: int = int(os.getenv("HT_MAX_HANDS", "2"))
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# -------------------------------------------------
# Tracking (HandTracker)
# -------------------------------------------------

@dataclass(frozen=True)
class Tracking:
    max_track_distance: float = float(os.getenv("HT_MAX_DISTANCE", "80.0"))
    max_missing_frames: int = int(os.getenv("HT_MAX_MISSING_FRAMES", "5"))


# -------------------------------------------------
# Smoothing (LandmarkSmoother)
# -------------------------------------------------

@dataclass(frozen=True)
class Smoothing:
    alpha: float = float(os.getenv("HT_SMOOTHING_ALPHA", "0.5"))


# -------------------------------------------------
# Objects (YOLO-World + ByteTrack)
# -------------------------------------------------

@dataclass
class Objects:
    enabled: bool = os.getenv("HT_OBJECTS", "1") not in ("0", "false", "False", "no")
    weights: str = os.getenv("HT_YOLO_WEIGHTS", "yolov8s-worldv2.pt")
    tracker: str = "bytetrack.yaml"
    min_confidence: float = float(os.getenv("HT_OBJECT_CONF", "0.20"))
    iou: float = float(os.getenv("HT_OBJECT_IOU", "0.45"))
    imgsz: int = int(os.getenv("HT_OBJECT_IMGSZ", "1280"))
    max_det: int = int(os.getenv("HT_OBJECT_MAX_DET", "100"))
    augment: bool = os.getenv("HT_OBJECT_AUGMENT", "1") not in ("0", "false", "False", "no")
    device: str | None = (
        os.getenv("HT_DEVICE")
        if os.getenv("HT_DEVICE") and os.getenv("HT_DEVICE") != "auto"
        else None
    )

    # Prompts partitioned into portable tools/objects and fixed surfaces.
    # YOLO-World uses these as open-vocabulary text prompts — specific,
    # concrete nouns give far better recall than generic category words.
    portable_classes: tuple[str, ...] = (
        # Hand tools
        "screwdriver",
        "hammer",
        "wrench",
        "pliers",
        "drill",
        "knife",
        "scissors",
        "cutter",
        "chisel",
        "file",
        "saw",
        "measuring tape",
        "ruler",
        # Electronics & assembly
        "soldering iron",
        "multimeter",
        "wire",
        "cable",
        "circuit board",
        "electronic component",
        # Everyday portable objects
        "pen",
        "pencil",
        "marker",
        "bottle",
        "cup",
        "mug",
        "phone",
        "remote control",
        "book",
        "paper",
        "box",
        "bag",
        # Generic fallbacks
        "handheld tool",
        "tool",
        "part",
        "component",
        "object",
    )
    fixed_classes: tuple[str, ...] = (
        # Surfaces
        "workbench",
        "desk",
        "table",
        "countertop",
        "floor",
        "shelf",
        "surface",
        # Machines / fixed equipment
        "machine",
        "monitor",
        "keyboard",
        "computer",
        "laptop",
    )
    _classes: tuple[str, ...] | None = None

    @property
    def classes(self) -> tuple[str, ...]:
        if self._classes is not None:
            return self._classes
        return self.portable_classes + self.fixed_classes

    @classes.setter
    def classes(self, val: tuple[str, ...] | list[str]) -> None:
        self._classes = tuple(val)


# -------------------------------------------------
# Contact Estimation & State Machine
# -------------------------------------------------

@dataclass
class Contact:
    # Tracking quality gate (TQS)
    min_tracking_quality: float = 0.40

    # Multi-cue distance thresholds (normalized by hand scale: wrist to middle MCP)
    tip_distance_max: float = 0.60
    palm_distance_max: float = 0.80
    min_overlap: float = 0.10

    # Cue weights and score threshold
    weight_tip: float = 0.50
    weight_palm: float = 0.30
    weight_overlap: float = 0.20
    score_threshold: float = 0.50

    # Large fixed surface rejection
    max_fixed_surface_area_ratio: float = 0.70

    # State machine temporal smoothing (in frames)
    enter_persistence_frames: int = 3
    unknown_enter_persistence_frames: int = 2
    unknown_exit_persistence_frames: int = 2
    min_state_duration_frames: int = 3

    # Gaps & absence
    max_gap_frames: int = 4

    # Events & evidence frames
    min_event_duration_s: float = 0.40
    evidence_unknown_min_s: float = 1.0
    evidence_frames_per_interval: int = 3


# -------------------------------------------------
# Master Configuration
# -------------------------------------------------

class Config:

    def __init__(self) -> None:
        self.paths = Paths()
        self.detection = Detection()
        self.tracking = Tracking()
        self.smoothing = Smoothing()
        self.objects = Objects()
        self.contact = Contact()

        self.expected_fps: int = int(os.getenv("HT_FPS", "30"))
        self.debug: bool = os.getenv("HT_DEBUG", "1") not in ("0", "false", "False", "no")
        self.save_debug_video: bool = (
            os.getenv("HT_SAVE_DEBUG_VIDEO", "1") not in ("0", "false", "False", "no")
        )
        self.json_indent: int = 2

    # Backward compatibility properties
    @property
    def enable_objects(self) -> bool:
        return self.objects.enabled

    @property
    def object_confidence(self) -> float:
        return self.objects.min_confidence

    @property
    def object_iou(self) -> float:
        return 0.45

    def ensure_dirs(self) -> None:
        self.paths.input_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.evidence_dir.mkdir(parents=True, exist_ok=True)

    def validate_inputs(self) -> None:
        if not self.paths.video.exists():
            raise FileNotFoundError(
                f"Input video not found:\n{self.paths.video}"
            )

        if not self.paths.model.exists():
            raise FileNotFoundError(
                f"MediaPipe hand model not found:\n{self.paths.model}"
            )

        if self.objects.enabled and not self.paths.yolo_model.exists():
            raise FileNotFoundError(
                f"YOLO model not found:\n{self.paths.yolo_model}"
            )


CONFIG = Config()

__all__ = [
    "Paths",
    "Detection",
    "Tracking",
    "Smoothing",
    "Objects",
    "Contact",
    "Config",
    "CONFIG",
]