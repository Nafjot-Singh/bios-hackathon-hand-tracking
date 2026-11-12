import pytest

from tests.helpers import make_hand
from tracker.smoother import LandmarkSmoother
from tracker.tracker import HandTracker

W = H = 100


def run(tracker, smoother, hands):
    tracks = tracker.update(hands, W, H)
    smoother.smooth_tracks(tracks)
    return tracks


def first_x(tracks):
    return tracks[0].hand.landmarks[0].x


def test_first_frame_is_unchanged_and_second_is_blended():
    tracker, smoother = HandTracker(), LandmarkSmoother(alpha=0.5, max_missed=1)
    assert first_x(run(tracker, smoother, [make_hand(0.2)])) == pytest.approx(0.2)
    assert first_x(run(tracker, smoother, [make_hand(0.3)])) == pytest.approx(0.25)


def test_frozen_detections_are_never_modified():
    tracker, smoother = HandTracker(), LandmarkSmoother(alpha=0.5)
    run(tracker, smoother, [make_hand(0.2)])
    raw = make_hand(0.3)
    tracks = run(tracker, smoother, [raw])
    assert raw.landmarks[0].x == pytest.approx(0.3)   # input untouched
    assert tracks[0].hand is not raw                   # a new object was assigned
    assert tracks[0].hand.score == raw.score and tracks[0].hand.handedness == raw.handedness


def test_short_gap_keeps_history_long_gap_resets_it():
    tracker, smoother = HandTracker(max_missing_frames=5), LandmarkSmoother(alpha=0.5, max_missed=1)
    run(tracker, smoother, [make_hand(0.2)])
    run(tracker, smoother, [make_hand(0.3)])
    run(tracker, smoother, [])                                                      # missing 1
    assert first_x(run(tracker, smoother, [make_hand(0.4)])) == pytest.approx(0.325)  # blended
    run(tracker, smoother, [])
    run(tracker, smoother, [])                                                      # missing 2 > 1
    assert first_x(run(tracker, smoother, [make_hand(0.9)])) == pytest.approx(0.9)    # fresh start


def test_history_of_expired_tracks_is_dropped():
    tracker, smoother = HandTracker(max_missing_frames=1), LandmarkSmoother()
    run(tracker, smoother, [make_hand(0.2)])
    for _ in range(3):
        run(tracker, smoother, [])
    assert smoother._history == {}


def test_invalid_arguments():
    for alpha in (0, 1.5):
        with pytest.raises(ValueError):
            LandmarkSmoother(alpha=alpha)
