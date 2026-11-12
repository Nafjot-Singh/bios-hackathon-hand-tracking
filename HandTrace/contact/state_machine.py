"""
state_machine.py

Temporal smoothing of per-frame contact labels into stable intervals.

Raw per-frame labels flicker (a hand touching a tool is "nothing" for a single
frame, a detection is missed, and so on). This module turns them into
intervals with hysteresis:

- A new state is accepted only after it lasts long enough:
    unknown                       -> unknown_enter_persistence_frames
    nothing / contact states      -> max(enter_persistence_frames, min_state_duration_frames)
- A run that is too short is absorbed into the previous state, and the
  boundary of an accepted state is back-dated to where its run began.
- Changing the touched object (different object ID) starts a new interval.

It works offline on a complete timeline, so it is deterministic and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from config import CONFIG, Contact


@dataclass
class Segment:
    """A run of identical labels; `start` and `end` are inclusive timeline indices."""

    state: str
    key: Hashable        # touched-object identity for contact states, else None
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def same_label(self, other: Segment) -> bool:
        return (self.state, self.key) == (other.state, other.key)


def required_frames(state: str, settings: Contact = CONFIG.contact) -> int:
    """How many consecutive frames a state needs before it is accepted."""
    if state == "unknown":
        return max(1, settings.unknown_enter_persistence_frames)
    return max(1, settings.enter_persistence_frames, settings.min_state_duration_frames)


def smooth(
    labels: Sequence[tuple[str, Hashable]],
    settings: Contact = CONFIG.contact,
) -> list[Segment]:
    """
    Turn per-frame (state, object key) labels into stable segments.

    Returns segments that cover the whole timeline without gaps or overlaps.
    """
    # 1. Run-length encode.
    runs: list[Segment] = []
    for index, (state, key) in enumerate(labels):
        if runs and (runs[-1].state, runs[-1].key) == (state, key):
            runs[-1].end = index
        else:
            runs.append(Segment(state, key, index, index))

    # 2. Absorb runs that are too short into the previous state.
    accepted: list[Segment] = []
    for run in runs:
        if accepted and accepted[-1].same_label(run):
            accepted[-1].end = run.end
        elif accepted and run.length < required_frames(run.state, settings):
            accepted[-1].end = run.end
        else:
            accepted.append(run)

    # 3. A too-short first segment has nothing before it: give it to the next one.
    if len(accepted) > 1 and accepted[0].length < required_frames(accepted[0].state, settings):
        accepted[1].start = accepted[0].start
        del accepted[0]

    # 4. Merge neighbours that became identical.
    merged: list[Segment] = []
    for segment in accepted:
        if merged and merged[-1].same_label(segment):
            merged[-1].end = segment.end
        else:
            merged.append(segment)
    return merged
