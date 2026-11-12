"""ObjectDetector with a stand-in for Ultralytics (the real package is not needed for tests)."""

import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest


class FakeTensor:
    def __init__(self, data):
        self._data = np.array(data)

    def cpu(self):
        return self

    def numpy(self):
        return self._data


class FakeYOLOWorld:
    prompts = None

    def __init__(self, weights):
        self.weights = weights

    def set_classes(self, classes):
        FakeYOLOWorld.prompts = list(classes)

    def track(self, image, **options):
        FakeYOLOWorld.options = options
        boxes = NS(
            xyxy=FakeTensor([[10, 20, 110, 220], [0, 0, 50, 50], [5, 5, 9, 9]]),
            conf=FakeTensor([0.9, 0.5, 0.8]),
            cls=FakeTensor([0, 5, 99]),
            id=FakeTensor([4, 8, 9]),
            __len__=lambda self: 3,
        )
        boxes = type("Boxes", (), {"xyxy": boxes.xyxy, "conf": boxes.conf, "cls": boxes.cls,
                                   "id": boxes.id, "__len__": lambda self: 3})()
        names = {0: "handheld tool", 5: "workbench", 99: "banana"}   # "banana" is not one of our prompts
        return [NS(boxes=boxes, names=names)]


@pytest.fixture()
def fake_ultralytics(monkeypatch):
    module = types.ModuleType("ultralytics")
    module.YOLOWorld = FakeYOLOWorld
    monkeypatch.setitem(sys.modules, "ultralytics", module)


def test_detections_are_converted_and_filtered(fake_ultralytics):
    from config import CONFIG
    from objects.object_detector import ObjectDetector

    with ObjectDetector() as detector:
        found = detector.detect(np.zeros((100, 100, 3), np.uint8))

    assert FakeYOLOWorld.prompts == list(CONFIG.objects.classes)
    assert FakeYOLOWorld.options["persist"] is True and FakeYOLOWorld.options["tracker"] == "bytetrack.yaml"
    assert [(o.label, o.object_id, o.kind) for o in found] == [("handheld tool", 4, "portable"), ("workbench", 8, "fixed")]
    assert found[0].bbox == (10.0, 20.0, 110.0, 220.0) and found[0].area == 100 * 200


def test_closed_detector_and_missing_ultralytics(fake_ultralytics, monkeypatch):
    from objects.object_detector import ObjectDetector

    detector = ObjectDetector()
    detector.close()
    with pytest.raises(RuntimeError):
        detector.detect(np.zeros((10, 10, 3), np.uint8))

    monkeypatch.setitem(sys.modules, "ultralytics", None)   # makes `import ultralytics` fail
    with pytest.raises(ImportError, match="pip install ultralytics"):
        ObjectDetector()
