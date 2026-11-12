import json

import numpy as np
import pytest

from tests.helpers import make_hand
from tracker.tracker import HandTracker


def test_visualizer_writes_a_video_and_validates_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from utils.visualization import Visualizer

    frame = np.zeros((120, 160, 3), np.uint8)
    tracks = HandTracker().update([make_hand(0.3), make_hand(0.7)], 160, 120)

    path = tmp_path / "debug.mp4"
    with Visualizer(160, 120, 30, path) as visualizer:
        for index in range(3):
            visualizer.write(frame, tracks, index)
        with pytest.raises(ValueError):
            visualizer.write(frame.astype(np.float32), tracks)
        with pytest.raises(ValueError):
            visualizer.write(np.zeros((10, 10, 3), np.uint8), tracks)
    assert (frame == 0).all()   # input frame is never drawn on

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()


def test_graphs_and_report_are_created(tmp_path):
    pytest.importorskip("matplotlib")
    from quality.graph import GraphGenerator
    from quality.metrics import MetricsCalculator
    from quality.report import ReportGenerator

    jsonl = tmp_path / "hands.jsonl"
    rows = [json.dumps({"frame": f, "track_id": 1, "handedness": "Left", "confidence": 0.9})
            for f in range(10)]
    jsonl.write_text("\n".join(rows) + "\n")

    metrics = MetricsCalculator(jsonl, total_frames=12).calculate()
    graphs = GraphGenerator(metrics, output_dir=tmp_path / "graphs", jsonl_path=jsonl).generate_all()
    assert {"hands_over_time", "track_timeline", "confidence"} <= set(graphs)
    assert all(p.stat().st_size > 0 for p in graphs.values())

    report = ReportGenerator(metrics, graphs, tmp_path / "report.html", source_name="a<b>.mp4").generate()
    page = report.read_text(encoding="utf-8")
    assert page.count("<img") == len(graphs)
    assert "a&lt;b&gt;.mp4" in page          # text is HTML-escaped
