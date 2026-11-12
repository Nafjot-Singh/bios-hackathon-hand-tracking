from tracker.matching import match_hands, match_points


def test_matching_does_not_depend_on_order():
    # Old one-at-a-time matching paired 0->6 and 10->-50; the best pairing is 10->6, 0->-50.
    result = match_points([(0, 0), (10, 0)], [(6, 0), (-50, 0)], max_distance=80)
    assert result.matches == {1: 0, 0: 1}
    assert not result.unmatched_previous and not result.unmatched_current


def test_pairs_beyond_max_distance_stay_unmatched():
    result = match_points([(0, 0)], [(500, 0)], max_distance=80)
    assert result.matches == {}
    assert result.unmatched_previous == [0]
    assert result.unmatched_current == [0]


def test_empty_inputs():
    assert match_points([], [(1, 1)]).unmatched_current == [0]
    assert match_points([(1, 1)], []).unmatched_previous == [0]


def test_match_hands_uses_pixel_centroids():
    from tests.helpers import make_hand

    previous = [make_hand(0.1), make_hand(0.8)]
    current = [make_hand(0.82), make_hand(0.12)]
    result = match_hands(previous, current, 100, 100, max_distance=80)
    assert result.matches == {0: 1, 1: 0}
