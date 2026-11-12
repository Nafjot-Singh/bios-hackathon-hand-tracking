import json

import pytest

from tests.helpers import make_hand
from tracker.tracker import HandTracker
from utils.json_writer import JSONWriter


def test_only_visible_tracks_are_written_as_valid_jsonl(tmp_path):
    tracker = HandTracker()
    path = tmp_path / "hands.jsonl"

    with JSONWriter(path, flush_every_frames=1) as writer:
        tracks = tracker.update([make_hand(0.2), make_hand(0.8)], 100, 100)
        assert writer.write_frame(0, 0, tracks) == 2
        tracks = tracker.update([make_hand(0.2)], 100, 100)   # track 2 only coasting
        assert writer.write_frame(1, 33, tracks) == 1
        assert writer.write_frame(2, 66, []) == 0

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["frame"], r["track_id"]) for r in records] == [(0, 1), (0, 2), (1, 1)]
    assert len(records[0]["landmarks"]) == 21
    assert set(records[0]) >= {"timestamp_ms", "handedness", "confidence", "centroid", "landmarks"}
    assert writer.frames_written == 3 and writer.records_written == 3


def test_writing_after_close_and_nan_are_rejected(tmp_path):
    writer = JSONWriter(tmp_path / "a.jsonl")
    writer.close()
    writer.close()  # safe twice
    with pytest.raises(ValueError):
        writer.write_frame(0, 0, [])

    bad = HandTracker().update([make_hand(float("nan"))], 100, 100)
    with JSONWriter(tmp_path / "b.jsonl") as writer, pytest.raises(ValueError):
        writer.write_frame(0, 0, bad)
