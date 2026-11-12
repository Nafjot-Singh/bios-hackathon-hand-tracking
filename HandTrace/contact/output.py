"""
output.py

Writes the contact stage results to disk.

- contacts.csv            one row per contact interval
- events_proposals.json   candidate events for review and naming
- evidence_frames/*.png   saved frames with the hands and objects drawn on them

Records are built from the Pydantic models in schemas.py, so they are
validated before they are written.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import cv2

from config import CONFIG
from contact.analyzer import Annotation, ContactResult
from contact.schemas import CONTACT_COLUMNS, ContactInterval, EventProposal
from utils.video_reader import VideoReader

log = logging.getLogger(__name__)

HAND_COLOR = (0, 255, 0)       # BGR
OBJECT_COLOR = (0, 165, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def write_contacts_csv(intervals: Sequence[ContactInterval], path: Path | str | None = None) -> Path:
    """Write contacts.csv. Empty values are blank cells."""
    path = Path(path or CONFIG.paths.contacts_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONTACT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for interval in intervals:
            row = interval.model_dump()
            row["t_start"] = f"{interval.t_start:.3f}"
            row["t_end"] = f"{interval.t_end:.3f}"
            row["evidence_frames"] = "|".join(str(n) for n in interval.evidence_frames)
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in CONTACT_COLUMNS})
    return path


def write_events_json(events: Sequence[EventProposal], path: Path | str | None = None) -> Path:
    """Write events_proposals.json (a list of event records)."""
    path = Path(path or CONFIG.paths.events_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.model_dump(mode="json") for e in events], indent=CONFIG.json_indent) + "\n",
        encoding="utf-8",
    )
    return path


def save_evidence_frames(
    video_path: Path | str,
    annotations: Mapping[int, Sequence[Annotation]],
    clip: str,
    out_dir: Path | str | None = None,
    fallback_fps: float = CONFIG.expected_fps,
) -> list[Path]:
    """
    Save annotated evidence frames as <clip>_<frame>.png.

    The video is read once, in order, up to the last frame needed (seeking
    is unreliable for some codecs).
    """
    if not annotations:
        return []

    out_dir = Path(out_dir or CONFIG.paths.evidence_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(annotations)
    last_needed = max(wanted)

    saved: list[Path] = []
    with VideoReader(video_path, fallback_fps) as reader:
        for frame in reader.frames():
            if frame.index > last_needed:
                break
            if frame.index not in wanted:
                continue

            image = frame.image.copy()
            for annotation in annotations[frame.index]:
                _draw_annotation(image, annotation)

            path = out_dir / f"{clip}_{frame.index}.png"
            if cv2.imwrite(str(path), image):
                saved.append(path)
            else:
                log.warning("Could not write evidence frame %s", path)

    missing = wanted - {int(p.stem.rsplit("_", 1)[1]) for p in saved}
    if missing:
        log.warning("%d evidence frame(s) could not be read from the video", len(missing))
    return saved


def _draw_annotation(image, annotation: Annotation) -> None:
    """Draw the hand box, the touched object box and a caption."""
    x0, y0, x1, y1 = (round(v) for v in annotation.hand_bbox)
    cv2.rectangle(image, (x0, y0), (x1, y1), HAND_COLOR, 2)

    caption = f"{annotation.hand}: {annotation.state}"
    if annotation.object_bbox is not None:
        ox0, oy0, ox1, oy1 = (round(v) for v in annotation.object_bbox)
        cv2.rectangle(image, (ox0, oy0), (ox1, oy1), OBJECT_COLOR, 2)
        caption += f" ({annotation.object_label})"

    origin = (x0, max(15, y0 - 6))
    cv2.putText(image, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 3, cv2.LINE_AA)
    cv2.putText(image, caption, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def save_all(result: ContactResult, video_path: Path | str, clip: str) -> dict[str, Path]:
    """Write contacts.csv, events_proposals.json and the evidence frames."""
    paths = {
        "contacts": write_contacts_csv(result.intervals),
        "events": write_events_json(result.events),
    }
    frames = save_evidence_frames(video_path, result.annotations, clip)
    if frames:
        paths["evidence_frames"] = CONFIG.paths.evidence_dir
    return paths
