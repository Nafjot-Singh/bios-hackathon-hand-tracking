"""
schemas.py

Frozen output schemas for the contact stage, validated with Pydantic.

Every record written to contacts.csv and events_proposals.json is built from
one of these models, so a malformed record fails loudly instead of being saved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ContactStateName = Literal["nothing", "portable_object", "fixed_surface", "unknown"]
UnknownReason = Literal["gap", "no_hand", "low_quality", "conflicting_cues"]

# Column order of contacts.csv. The first nine are the shared contract;
# `unknown_reason` is an additive last column.
CONTACT_COLUMNS = (
    "clip", "t_start", "t_end", "hand", "state", "object_name", "object_id",
    "persistence_frames", "evidence_frames", "unknown_reason",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ContactInterval(_Frozen):
    """One hand in one contact state over a time interval."""

    clip: str
    t_start: float = Field(ge=0)
    t_end: float = Field(ge=0)
    hand: str                                  # e.g. "left_0", "right_0"
    state: ContactStateName
    object_name: str | None = None
    object_id: int | None = None
    persistence_frames: int = Field(ge=1)
    evidence_frames: tuple[int, ...] = ()      # frame indices of saved evidence images
    unknown_reason: UnknownReason | None = None

    @model_validator(mode="after")
    def _check(self) -> ContactInterval:
        if self.t_end < self.t_start:
            raise ValueError("t_end must not be before t_start")
        if self.state == "unknown" and self.unknown_reason is None:
            raise ValueError("an unknown interval needs an unknown_reason")
        if self.state in ("portable_object", "fixed_surface") and self.object_name is None:
            raise ValueError(f"state {self.state} needs an object_name")
        return self


class EventProposal(_Frozen):
    """A candidate event for human review and naming."""

    clip_id: str
    event_id: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(ge=0)
    hand_track_id: int
    hand_side: str
    object_id: int | None = None
    object_label: str
    contact_state: ContactStateName
    tracking_quality: float = Field(ge=0, le=1)
    evidence_frames: tuple[str, ...] = ()      # paths relative to the outputs folder
    review_status: Literal["pending", "accepted", "rejected"] = "pending"
    action: str | None = None                  # filled in later by event naming
    trigger_reason: str

    @model_validator(mode="after")
    def _check(self) -> EventProposal:
        if self.end_sec < self.start_sec:
            raise ValueError("end_sec must not be before start_sec")
        return self
