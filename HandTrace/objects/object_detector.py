"""
YOLO Object Detector
--------------------

Uses Ultralytics YOLO (works with YOLO-World and standard YOLO models) to
detect and optionally track objects in each frame.

Save this file as:  objects/object_detector.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    track_id: int | None = None
    # "portable" | "fixed" | "unknown"  – set by the detector based on config class lists
    kind: str = "unknown"

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def label(self) -> str:
        """Human-readable label (alias for class_name)."""
        return self.class_name

    @property
    def object_id(self) -> int | None:
        """Stable tracking ID (alias for track_id)."""
        return self.track_id

    @property
    def area(self) -> float:
        """Pixel area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))


class ObjectDetector:
    def __init__(self) -> None:
        cfg = CONFIG.objects

        self.model = YOLO(str(CONFIG.paths.yolo_model))
        self.conf = cfg.min_confidence
        self.imgsz = cfg.imgsz
        self.iou = getattr(cfg, "iou", 0.45)
        self.max_det = getattr(cfg, "max_det", 100)
        self.augment = getattr(cfg, "augment", True)
        self.device = getattr(cfg, "device", None)
        self.tracker = getattr(cfg, "tracker", "bytetrack.yaml")
        self._track_failed = False

        # Sets used to classify detections as "portable" or "fixed" (lowercase).
        # Both exact match and substring match are checked so that YOLO-World
        # variants (e.g. "power drill" matching "drill") are caught.
        self._portable_classes: frozenset[str] = frozenset(
            c.lower() for c in getattr(cfg, "portable_classes", [])
        )
        self._fixed_classes: frozenset[str] = frozenset(
            c.lower() for c in getattr(cfg, "fixed_classes", [])
        )

        # Which classes to look for.
        # YOLO-World: set_classes() changes the vocabulary.
        # Standard YOLO: filter by class index inside predict().
        self.class_ids: list[int] | None = None
        wanted = list(getattr(cfg, "classes", []) or [])

        if wanted:
            try:
                self.model.set_classes(wanted)
                logger.info("YOLO-World classes set: %s", wanted)
            except Exception as exc:
                logger.info("set_classes not available (%s); using index filter", exc)
                self.class_ids = self._ids_from_names(wanted)

        self._warmup()

    # Allows `with ObjectDetector() as det:` and ExitStack.enter_context(...)
    def __enter__(self) -> "ObjectDetector":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    # ------------------------------------------------------------------ #
    def _ids_from_names(self, wanted: list[str]) -> list[int] | None:
        name_to_id = {str(n).lower(): i for i, n in self.model.names.items()}
        ids, missing = [], []
        for name in wanted:
            idx = name_to_id.get(name.lower())
            if idx is None:
                missing.append(name)
            else:
                ids.append(idx)
        if missing:
            logger.warning("Classes not in model vocabulary, ignored: %s", missing)
        return ids or None

    def _warmup(self) -> None:
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._run(dummy, track=False)
        except Exception as exc:
            logger.debug("Warmup skipped: %s", exc)

    def _run(self, frame: np.ndarray, track: bool):
        kwargs = dict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.class_ids,
            device=self.device,
            augment=self.augment,
            verbose=False,
        )
        if track and not self._track_failed:
            try:
                return self.model.track(persist=True, tracker=self.tracker, **kwargs)
            except Exception as exc:
                # Tracking needs extra packages (e.g. lap). Fall back, don't crash.
                logger.warning("Tracking unavailable (%s); using plain detection", exc)
                self._track_failed = True
        return self.model.predict(**kwargs)

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray, track: bool = False) -> list[ObjectDetection]:
        """Detect objects in a BGR frame. Use track=True to fill track_id."""
        if frame is None or frame.size == 0:
            return []

        results = self._run(frame, track)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = [None] * len(boxes)

        h, w = frame.shape[:2]
        names = result.names
        detections: list[ObjectDetection] = []

        for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, clss, ids):
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = names[cls]
            cn_lower = class_name.lower()
            # Exact match first, then substring — handles YOLO-World variants
            # like "power drill" matching the prompt "drill".
            if cn_lower in self._portable_classes or any(
                p in cn_lower or cn_lower in p for p in self._portable_classes
            ):
                kind = "portable"
            elif cn_lower in self._fixed_classes or any(
                f in cn_lower or cn_lower in f for f in self._fixed_classes
            ):
                kind = "fixed"
            else:
                kind = "unknown"

            detections.append(
                ObjectDetection(
                    class_name=class_name,
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    track_id=int(tid) if tid is not None else None,
                    kind=kind,
                )
            )

        return detections