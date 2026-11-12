"""
YOLO Object Detector
--------------------

Uses Ultralytics YOLO to detect (and optionally track) objects in each frame.

Returns a list of ObjectDetection instances that the rest of the pipeline
can use directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from config import CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels
    track_id: int | None = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class ObjectDetector:
    def __init__(self) -> None:
        cfg = CONFIG.objects

        self.model = YOLO(str(CONFIG.paths.yolo_model))
        self.conf = cfg.min_confidence
        self.imgsz = cfg.imgsz
        self.iou = getattr(cfg, "iou", 0.7)
        self.max_det = getattr(cfg, "max_det", 100)
        self.device = getattr(cfg, "device", None)  # e.g. "cuda:0", "cpu", None = auto
        self.half = getattr(cfg, "half", False)  # FP16, only useful on GPU

        # Resolve which classes to keep.
        # - YOLO-World / open-vocab models: set_classes() changes the vocabulary.
        # - Standard COCO models: set_classes doesn't exist, so filter by class
        #   index in predict() instead (done on the model, not in Python).
        self.class_ids: list[int] | None = None
        wanted = list(getattr(cfg, "classes", []) or [])

        if wanted:
            if hasattr(self.model, "set_classes"):
                try:
                    self.model.set_classes(wanted)
                except Exception as exc:
                    logger.warning("set_classes failed (%s); falling back to index filter", exc)
                    self.class_ids = self._ids_from_names(wanted)
            else:
                self.class_ids = self._ids_from_names(wanted)

        self._warmup()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ids_from_names(self, wanted: list[str]) -> list[int] | None:
        name_to_id = {n.lower(): i for i, n in self.model.names.items()}
        ids, missing = [], []
        for name in wanted:
            idx = name_to_id.get(name.lower())
            (ids if idx is not None else missing).append(idx if idx is not None else name)
        if missing:
            logger.warning("Classes not in model vocabulary, ignored: %s", missing)
        return ids or None

    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow."""
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
            half=self.half,
            verbose=False,
        )
        if track:
            # persist=True keeps tracker state between consecutive frames
            return self.model.track(persist=True, **kwargs)
        return self.model.predict(**kwargs)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray, track: bool = False) -> list[ObjectDetection]:
        """Detect objects in a BGR frame. Set track=True to fill track_id."""
        if frame is None or frame.size == 0:
            return []

        results = self._run(frame, track)
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Move everything to CPU/numpy once, instead of per-box .item() calls.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if track and boxes.id is not None
            else [None] * len(boxes)
        )

        h, w = frame.shape[:2]
        names = result.names
        detections: list[ObjectDetection] = []

        for (x1, y1, x2, y2), conf, cls, tid in zip(xyxy, confs, clss, ids):
            # Clamp to frame so downstream cropping never goes out of bounds.
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w, x2)))
            y2 = int(max(0, min(h, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                ObjectDetection(
                    class_name=names[cls],
                    confidence=float(conf),
                    bbox=(x1, y1, x2, y2),
                    track_id=int(tid) if tid is not None else None,
                )
            )

        return detections