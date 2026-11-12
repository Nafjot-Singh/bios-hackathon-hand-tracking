import csv
import json

import numpy as np
import pytest

from config import Contact
from contact.analyzer import ContactAnalyzer, HandNamer
from contact.cues import classify_frame, overlap_share, point_to_box_distance, tracking_quality
from contact.schemas import ContactInterval, EventProposal
from contact.state_machine import smooth
from objects.object_detector import ObjectDetection
from tests.helpers import Landmark, HandDetection, make_hand
from tracker.tracker import HandTracker

W, H = 1000, 1000


def hand_at(cx, cy, size=0.1, score=0.95):
    """A hand with its wrist at (cx, cy) and fingers pointing up; fingertips ~`size` above."""
    pts = [Landmark(cx, cy, 0.0) for _ in range(21)]
    pts[9] = Landmark(cx, cy - size * 0.6, 0.0)                       # middle-finger base: sets hand scale
    for tip, dx in zip((4, 8, 12, 16, 20), (-0.03, -0.015, 0.0, 0.015, 0.03)):
        pts[tip] = Landmark(cx + dx, cy - size, 0.0)
    for base, dx in zip((5, 13, 17), (-0.015, 0.015, 0.03)):
        pts[base] = Landmark(cx + dx, cy - size * 0.6, 0.0)
    return HandDetection("Left", score, tuple(pts))


def obj(x0, y0, x1, y1, kind="portable", label="handheld tool", object_id=7):
    return ObjectDetection(object_id, label, 0.9, (x0, y0, x1, y1), kind)


# ---------------- geometry and cues ----------------

def test_geometry_helpers():
    assert point_to_box_distance((5, 5), (0, 0, 10, 10)) == 0
    assert point_to_box_distance((13, 14), (0, 0, 10, 10)) == 5
    assert overlap_share((0, 0, 10, 10), (5, 0, 20, 10)) == pytest.approx(0.5)
    assert overlap_share((0, 0, 0, 0), (0, 0, 5, 5)) == 0


def test_tracking_quality_drops_when_cut_off_or_unsure():
    assert tracking_quality(hand_at(0.5, 0.8)) == pytest.approx(0.95)
    assert tracking_quality(hand_at(0.5, 0.8, score=0.3)) < 0.4
    assert tracking_quality(hand_at(0.5, 1.5)) == 0          # every landmark outside the image


def test_fingertips_on_a_portable_object_is_contact():
    hand = hand_at(0.5, 0.8)            # fingertips at y = 700 px
    result = classify_frame(hand, [obj(400, 600, 600, 720)], W, H)
    assert result.state == "portable_object" and result.obj.label == "handheld tool"


def test_overlap_alone_is_never_contact():
    hand = hand_at(0.5, 0.8)
    far_big = obj(0, 0, 1000, 300)      # nowhere near the fingertips
    assert classify_frame(hand, [far_big], W, H).state == "nothing"


def test_portable_beats_fixed_and_nothing_when_no_objects():
    hand = hand_at(0.5, 0.8)
    both = [obj(300, 690, 700, 760, "fixed", "workbench", 1), obj(400, 600, 600, 720)]
    assert classify_frame(hand, both, W, H).state == "portable_object"
    assert classify_frame(hand, [obj(300, 690, 700, 760, "fixed", "workbench", 1)], W, H).state == "fixed_surface"
    assert classify_frame(hand, [], W, H).state == "nothing"


def test_full_frame_fixed_surface_is_rejected_as_conflicting():
    hand = hand_at(0.5, 0.8)
    result = classify_frame(hand, [obj(0, 0, 1000, 960, "fixed", "workbench", 3)], W, H)
    assert (result.state, result.reason) == ("unknown", "conflicting_cues")


def test_low_quality_hand_is_unknown():
    result = classify_frame(hand_at(0.5, 0.8, score=0.2), [obj(400, 600, 600, 720)], W, H)
    assert (result.state, result.reason) == ("unknown", "low_quality")


# ---------------- state machine ----------------

def labels(text):
    key = {"N": ("nothing", None), "P": ("portable_object", 7), "U": ("unknown", None)}
    return [key[c] for c in text]


def spans(text, settings=Contact()):
    return [(s.state, s.start, s.end) for s in smooth(labels(text), settings)]


def test_flicker_is_absorbed():
    assert spans("PPPPNPPPP") == [("portable_object", 0, 8)]
    assert spans("NNNNPPPPNNNN") == [("nothing", 0, 3), ("portable_object", 4, 7), ("nothing", 8, 11)]


def test_short_unknown_is_bridged_but_two_frames_are_reported():
    assert spans("PPPPUPPPP") == [("portable_object", 0, 8)]
    assert spans("PPPPUUPPPP") == [("portable_object", 0, 3), ("unknown", 4, 5), ("portable_object", 6, 9)]


def test_boundaries_are_back_dated_and_first_short_run_joins_next():
    assert spans("NNNPPP")[1] == ("portable_object", 3, 5)
    assert spans("PNNNN") == [("nothing", 0, 4)]


def test_changing_object_starts_a_new_interval():
    result = smooth([("portable_object", 7)] * 4 + [("portable_object", 8)] * 4)
    assert [(s.key, s.start, s.end) for s in result] == [(7, 0, 3), (8, 4, 7)]


# ---------------- analyzer end to end ----------------

def run_clip(frames_spec, fps=10.0):
    """frames_spec: list of 'tool' / 'none' / 'miss' per frame."""
    tracker, analyzer = HandTracker(max_missing_frames=10), ContactAnalyzer("clip", fps)  # keep the track alive across gaps
    tool = [obj(400, 600, 600, 720)]
    for index, kind in enumerate(frames_spec):
        hands = [] if kind == "miss" else [hand_at(0.5, 0.8)]
        tracks = tracker.update(hands, W, H)
        analyzer.observe(index, tracker.visible_tracks, tool if kind == "tool" else [], W, H)
    return analyzer.finalize()


def test_intervals_events_and_unknown_gap():
    result = run_clip(["none"] * 5 + ["tool"] * 10 + ["miss"] * 3 + ["none"] * 5)
    assert [i.state for i in result.intervals] == ["nothing", "portable_object", "unknown", "nothing"]
    grasp = result.intervals[1]
    assert (grasp.hand, grasp.object_name, grasp.persistence_frames) == ("left_0", "handheld tool", 10)
    assert grasp.t_start == pytest.approx(0.5) and grasp.t_end == pytest.approx(1.5)
    assert result.intervals[2].unknown_reason == "gap"                      # 3 missing frames <= max_gap_frames
    assert len(result.events) == 1
    event = result.events[0]
    assert (event.event_id, event.hand_side, event.review_status, event.action) == ("clip_E001", "left", "pending", None)
    assert event.evidence_frames == tuple(f"evidence_frames/clip_{f}.png" for f in grasp.evidence_frames)
    # intervals tile the timeline
    for a, b in zip(result.intervals, result.intervals[1:]):
        assert a.t_end == pytest.approx(b.t_start)


def test_long_absence_is_no_hand_and_short_contact_is_not_an_event():
    result = run_clip(["none"] * 4 + ["miss"] * 8 + ["none"] * 4 + ["tool"] * 3 + ["none"] * 4)
    unknown = [i for i in result.intervals if i.state == "unknown"][0]
    assert unknown.unknown_reason == "no_hand"
    assert result.events == []                                              # 0.3 s < min_event_duration_s


def test_hand_namer_and_validation():
    namer = HandNamer()
    assert [namer.name(5, "Left"), namer.name(6, "Right"), namer.name(9, "Left"), namer.name(5, "Right")] == \
           ["left_0", "right_0", "left_1", "left_0"]
    with pytest.raises(ValueError):
        ContactInterval(clip="c", t_start=2, t_end=1, hand="left_0", state="nothing", persistence_frames=1)
    with pytest.raises(ValueError):
        ContactInterval(clip="c", t_start=0, t_end=1, hand="left_0", state="unknown", persistence_frames=1)
    with pytest.raises(ValueError):
        EventProposal(clip_id="c", event_id="e", start_sec=0, end_sec=1, hand_track_id=1, hand_side="left",
                      object_label="x", contact_state="portable_object", tracking_quality=2.0,
                      trigger_reason="t")


def test_outputs_are_written(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from contact.output import save_all, write_contacts_csv, write_events_json
    from config import CONFIG
    import contact.output as output

    result = run_clip(["none"] * 5 + ["tool"] * 10 + ["miss"] * 3 + ["none"] * 5)

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    for _ in range(25):
        writer.write(np.zeros((H, W, 3), np.uint8))
    writer.release()

    csv_path = write_contacts_csv(result.intervals, tmp_path / "contacts.csv")
    rows = list(csv.DictReader(csv_path.open()))
    assert list(rows[0]) == ["clip", "t_start", "t_end", "hand", "state", "object_name", "object_id",
                             "persistence_frames", "evidence_frames", "unknown_reason"]
    assert rows[1]["state"] == "portable_object" and rows[1]["object_id"] == "7" and "|" in rows[1]["evidence_frames"]
    assert rows[0]["object_name"] == "" and rows[2]["unknown_reason"] == "gap"

    events = json.loads(write_events_json(result.events, tmp_path / "events.json").read_text())
    assert set(events[0]) == {"clip_id", "event_id", "start_sec", "end_sec", "hand_track_id", "hand_side",
                              "object_id", "object_label", "contact_state", "tracking_quality",
                              "evidence_frames", "review_status", "action", "trigger_reason"}

    saved = output.save_evidence_frames(video, result.annotations, "clip", tmp_path / "evidence")
    assert saved and all(p.exists() and p.name.startswith("clip_") for p in saved)
