"""
analyzer.py

Collects per-frame hand-object observations and turns them into contact
intervals and event proposals.

Responsibilities:
- Name hands (left_0, right_0, ...) so the contact files use stable labels
- Classify every visible hand in every frame (cues.py)
- After the video: smooth labels into intervals (state_machine.py),
  describe unknown intervals, and propose events

This module DOES NOT run detectors or trackers, and does not write files
(see output.py).

Usage:
    analyzer = ContactAnalyzer(clip="demo", fps=30.0)
    for frame in ...:
        analyzer.observe(frame.index, visible_tracks, objects, width, height)
    result = analyzer.finalize()
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from config import CONFIG, Contact
from contact.cues import FrameContact, classify_frame
from contact.schemas import ContactInterval, EventProposal
from contact.state_machine import smooth
from detector.landmark_utils import compute_bbox_px

if TYPE_CHECKING:  # type hints only
    from objects.object_detector import ObjectDetection
    from tracker.tracker import Track

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Observation:
    """What was seen for one hand in one frame."""

    contact: FrameContact
    hand_bbox: Box


@dataclass(frozen=True)
class Annotation:
    """What to draw on an evidence frame for one hand."""

    hand: str
    state: str
    hand_bbox: Box
    object_label: str | None = None
    object_bbox: Box | None = None


@dataclass
class ContactResult:
    """Everything the contact stage produced."""

    intervals: list[ContactInterval] = field(default_factory=list)
    events: list[EventProposal] = field(default_factory=list)
    annotations: dict[int, list[Annotation]] = field(default_factory=dict)  # frame -> drawings for evidence images
    flicker_per_minute: dict[str, float] = field(default_factory=dict)      # hand -> intervals per active minute

    @property
    def unknown_share(self) -> float:
        if not self.intervals:
            return 0.0
        return sum(i.state == "unknown" for i in self.intervals) / len(self.intervals)

    def summary(self) -> str:
        unknown = sum(i.state == "unknown" for i in self.intervals)
        lines = [
            f"Contact: {len(self.intervals)} intervals, {len(self.events)} event proposals, "
            f"{unknown} unknown ({self.unknown_share:.1%})",
        ]
        for hand, rate in sorted(self.flicker_per_minute.items()):
            lines.append(f"  {hand:<10} flicker {rate:.1f} intervals/min")
        return "\n".join(lines)


class HandNamer:
    """
    Gives each hand track a readable name: <side>_<n>, e.g. left_0, right_0.

    The side is the handedness seen when the track first appears. A later
    track on the same side (for example after an ID switch) becomes left_1.
    """

    def __init__(self) -> None:
        self._names: dict[int, str] = {}
        self._counts: Counter[str] = Counter()

    def name(self, track_id: int, handedness: str) -> str:
        if track_id not in self._names:
            side = handedness.lower() or "unknown"
            self._names[track_id] = f"{side}_{self._counts[side]}"
            self._counts[side] += 1
        return self._names[track_id]


class ContactAnalyzer:
    """Builds contact intervals and event proposals for one clip."""

    def __init__(self, clip: str, fps: float, settings: Contact = CONFIG.contact) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.clip = clip
        self.fps = fps
        self.settings = settings

        self._namer = HandNamer()
        self._frames: defaultdict[str, dict[int, Observation]] = defaultdict(dict)
        self._track_ids: dict[str, int] = {}

    # -----------------------------------------------------
    # Collecting
    # -----------------------------------------------------

    def observe(
        self,
        frame_index: int,
        tracks: Sequence[Track],
        objects: Sequence[ObjectDetection],
        width: int,
        height: int,
    ) -> None:
        """Classify every visible hand in this frame."""
        for track in tracks:
            if not track.updated_this_frame:
                continue
            name = self._namer.name(track.track_id, track.hand.handedness)
            self._track_ids.setdefault(name, track.track_id)

            contact = classify_frame(track.hand, objects, width, height, self.settings)
            self._frames[name][frame_index] = Observation(
                contact, compute_bbox_px(track.hand, width, height)
            )

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    def finalize(self) -> ContactResult:
        """Smooth everything collected so far into intervals and events."""
        result = ContactResult()
        pending_events: list[dict] = []

        for hand in sorted(self._frames):
            observed = self._frames[hand]
            first, last = min(observed), max(observed)

            labels, reasons = self._timeline(observed, first, last)
            segments = smooth(labels, self.settings)

            active_minutes = (last - first + 1) / self.fps / 60.0
            result.flicker_per_minute[hand] = len(segments) / active_minutes if active_minutes else 0.0

            for segment in segments:
                frames = list(range(first + segment.start, first + segment.end + 1))
                seen = [observed[f] for f in frames if f in observed]

                object_name, object_id = self._dominant_object(seen)
                state = segment.state
                reason = self._dominant_reason(reasons[segment.start: segment.end + 1]) if state == "unknown" else None

                t_start = round(frames[0] / self.fps, 3)
                t_end = round((frames[-1] + 1) / self.fps, 3)   # end-exclusive, so intervals tile
                is_event = (
                    state == "portable_object"
                    and t_end - t_start >= self.settings.min_event_duration_s
                )
                wants_evidence = is_event or (
                    state == "unknown" and t_end - t_start >= self.settings.evidence_unknown_min_s
                )
                evidence = self._pick_frames(frames) if wants_evidence else ()

                result.intervals.append(
                    ContactInterval(
                        clip=self.clip, t_start=t_start, t_end=t_end, hand=hand, state=state,
                        object_name=object_name if state in ("portable_object", "fixed_surface") else None,
                        object_id=object_id if state in ("portable_object", "fixed_surface") else None,
                        persistence_frames=len(frames), evidence_frames=tuple(evidence),
                        unknown_reason=reason,
                    )
                )

                for f in evidence:
                    if f in observed:
                        result.annotations.setdefault(f, []).append(
                            self._annotation(hand, state, observed[f])
                        )
                    else:
                        result.annotations.setdefault(f, [])

                if is_event:
                    pending_events.append(
                        {
                            "hand": hand, "t_start": t_start, "t_end": t_end,
                            "object_name": object_name, "object_id": object_id,
                            "quality": sum(o.contact.tracking_quality for o in seen) / len(seen) if seen else 0.0,
                            "evidence": evidence,
                        }
                    )

        result.intervals.sort(key=lambda i: (i.t_start, i.hand))
        result.events = self._build_events(pending_events)
        return result

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    def _timeline(
        self, observed: dict[int, Observation], first: int, last: int
    ) -> tuple[list[tuple[str, object]], list[str | None]]:
        """
        Per-frame labels from a hand's first to last frame.

        Frames where the hand was not detected are `unknown`: reason `gap`
        for a short run (<= max_gap_frames), `no_hand` for a longer one.
        Nothing is invented for them.
        """
        labels: list[tuple[str, object]] = []
        reasons: list[str | None] = []
        frame = first
        while frame <= last:
            if frame in observed:
                contact = observed[frame].contact
                key = None
                if contact.obj is not None:
                    key = contact.obj.object_id if contact.obj.object_id is not None else contact.obj.label
                labels.append((contact.state, key))
                reasons.append(contact.reason)
                frame += 1
                continue
            run_end = frame
            while run_end + 1 <= last and (run_end + 1) not in observed:
                run_end += 1
            reason = "gap" if run_end - frame + 1 <= self.settings.max_gap_frames else "no_hand"
            for _ in range(frame, run_end + 1):
                labels.append(("unknown", None))
                reasons.append(reason)
            frame = run_end + 1
        return labels, reasons

    @staticmethod
    def _dominant_object(seen: Sequence[Observation]) -> tuple[str | None, int | None]:
        labels = Counter(o.contact.obj.label for o in seen if o.contact.obj)
        ids = Counter(o.contact.obj.object_id for o in seen
                      if o.contact.obj and o.contact.obj.object_id is not None)
        return (labels.most_common(1)[0][0] if labels else None,
                ids.most_common(1)[0][0] if ids else None)

    @staticmethod
    def _dominant_reason(reasons: Sequence[str | None]) -> str:
        counts = Counter(r for r in reasons if r)
        return counts.most_common(1)[0][0] if counts else "no_hand"

    def _pick_frames(self, frames: Sequence[int]) -> tuple[int, ...]:
        """Evenly spaced frames across an interval (first, middle, last, ...)."""
        count = min(self.settings.evidence_frames_per_interval, len(frames))
        if count <= 0:
            return ()
        if count == 1:
            return (frames[0],)
        step = (len(frames) - 1) / (count - 1)
        return tuple(sorted({frames[round(i * step)] for i in range(count)}))

    @staticmethod
    def _annotation(hand: str, state: str, observation: Observation) -> Annotation:
        obj = observation.contact.obj
        return Annotation(
            hand=hand, state=state, hand_bbox=observation.hand_bbox,
            object_label=obj.label if obj else None,
            object_bbox=obj.bbox if obj else None,
        )

    def _build_events(self, pending: list[dict]) -> list[EventProposal]:
        pending.sort(key=lambda e: (e["t_start"], e["hand"]))
        events: list[EventProposal] = []
        for number, item in enumerate(pending, start=1):
            hand = item["hand"]
            events.append(
                EventProposal(
                    clip_id=self.clip,
                    event_id=f"{self.clip}_E{number:03d}",
                    start_sec=item["t_start"], end_sec=item["t_end"],
                    hand_track_id=self._track_ids[hand],
                    hand_side=hand.rsplit("_", 1)[0],
                    object_id=item["object_id"], object_label=item["object_name"] or "unknown",
                    contact_state="portable_object",
                    tracking_quality=round(min(1.0, max(0.0, item["quality"])), 4),
                    evidence_frames=tuple(f"evidence_frames/{self.clip}_{f}.png" for f in item["evidence"]),
                    trigger_reason="sustained_portable_object_contact",
                )
            )
        return events
