from tests.helpers import make_hand
from tracker.tracker import HandTracker

W = H = 100


def ids(tracks):
    return [t.track_id for t in tracks]


def test_ids_persist_even_if_detection_order_changes():
    tracker = HandTracker(max_missing_frames=2)
    tracker.update([make_hand(0.2), make_hand(0.8)], W, H)
    tracks = tracker.update([make_hand(0.82), make_hand(0.21)], W, H)
    assert ids(tracks) == [1, 2]
    assert all(t.updated_this_frame for t in tracks)


def test_hand_returning_before_timeout_keeps_its_id():
    tracker = HandTracker(max_missing_frames=2)
    tracker.update([make_hand(0.2), make_hand(0.8)], W, H)
    tracks = tracker.update([make_hand(0.2)], W, H)
    assert ids(tracks) == [1, 2] and not tracks[1].updated_this_frame
    assert ids(tracker.visible_tracks) == [1]
    tracks = tracker.update([make_hand(0.2), make_hand(0.8)], W, H)
    assert ids(tracks) == [1, 2] and tracks[1].missing_frames == 0


def test_expired_track_is_deleted_and_ids_are_never_reused():
    tracker = HandTracker(max_missing_frames=2)
    tracker.update([make_hand(0.2), make_hand(0.8)], W, H)
    for _ in range(3):  # track 2 missing 3 frames > 2
        tracks = tracker.update([make_hand(0.2)], W, H)
    assert ids(tracks) == [1]
    tracks = tracker.update([make_hand(0.2), make_hand(0.8)], W, H)
    assert ids(tracks) == [1, 3]


def test_age_counts_every_frame_and_reset_restarts_ids():
    tracker = HandTracker()
    tracker.update([make_hand(0.2)], W, H)
    track = tracker.update([], W, H)[0]
    assert track.age == 2 and track.missing_frames == 1
    tracker.reset()
    assert tracker.active_tracks == []
    assert ids(tracker.update([make_hand(0.5)], W, H)) == [1]


def test_invalid_arguments():
    import pytest

    with pytest.raises(ValueError):
        HandTracker(max_track_distance=0)
    with pytest.raises(ValueError):
        HandTracker().update([], 0, 100)
