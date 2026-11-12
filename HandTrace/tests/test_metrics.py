import json

import pytest

from quality.metrics import MetricsCalculator


def build_file(path):
    rows = []

    def add(frame, track, hand, conf=0.95):
        rows.append(json.dumps({"frame": frame, "track_id": track, "handedness": hand,
                                "confidence": conf, "landmarks": []}))

    for f in range(0, 20):
        add(f, 1, "Left")                                    # left hand, ID 1
    for f in range(25, 40):
        add(f, 5, "Left")                                    # same hand, new ID -> switch
    for f in list(range(0, 10)) + list(range(18, 30)):
        add(f, 2, "Right", 0.7)                              # 8-frame disappearance
    path.write_text("\n".join(rows) + "\n" + '{"frame": 99, "track_id"')  # truncated last line


def test_metrics(tmp_path):
    path = tmp_path / "hands.jsonl"
    build_file(path)
    m = MetricsCalculator(path, total_frames=100).compute()

    assert m.total_frames == 100 and not m.total_frames_is_estimate
    assert m.skipped_lines == 1
    assert m.unique_tracks == 3
    assert m.longest_disappearance == 8 and m.longest_disappearance_track == 2
    assert m.tracks_with_gaps == 1 and m.total_gap_frames == 8
    assert m.id_switches == 1
    assert (m.id_switch_events[0]["from_track"], m.id_switch_events[0]["to_track"]) == (1, 5)
    assert m.low_confidence_records == 22   # track 2 records below 0.8


def test_total_frames_estimate_and_validation(tmp_path):
    path = tmp_path / "hands.jsonl"
    build_file(path)
    estimated = MetricsCalculator(path).compute()
    assert estimated.total_frames_is_estimate and estimated.total_frames == 40
    with pytest.raises(ValueError):
        MetricsCalculator(path, total_frames=5).compute()


def test_empty_file_and_save(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    calc = MetricsCalculator(path, total_frames=10)
    assert calc.calculate()["records"] == 0
    saved = calc.save(tmp_path / "metrics.json")
    assert json.loads(saved.read_text())["total_frames"] == 10
