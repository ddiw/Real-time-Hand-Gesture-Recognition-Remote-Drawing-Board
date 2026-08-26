import pytest

from app.geometry import NUM_LANDMARKS, Point
from app.one_euro_filter import HandLandmarksFilter


def _make_landmarks(x: float, y: float) -> list[Point]:
    return [Point(x, y, 0.0) for _ in range(NUM_LANDMARKS)]


def test_filter_first_call_passes_through_unchanged():
    f = HandLandmarksFilter()
    out = f.apply(_make_landmarks(0.5, 0.5), timestamp_ms=0)
    assert out[0].x == pytest.approx(0.5)
    assert out[0].y == pytest.approx(0.5)


def test_filter_smooths_a_single_frame_jitter_spike():
    f = HandLandmarksFilter(min_cutoff=1.0, beta=0.0)
    f.apply(_make_landmarks(0.5, 0.5), timestamp_ms=0)
    f.apply(_make_landmarks(0.5, 0.5), timestamp_ms=33)

    spiked = f.apply(_make_landmarks(0.9, 0.5), timestamp_ms=66)
    assert 0.5 < spiked[0].x < 0.9


def test_filter_converges_to_a_sustained_step():
    f = HandLandmarksFilter(min_cutoff=1.0, beta=0.0)
    t = 0
    for _ in range(60):
        f.apply(_make_landmarks(0.2, 0.2), timestamp_ms=t)
        t += 33

    for _ in range(60):
        result = f.apply(_make_landmarks(0.8, 0.2), timestamp_ms=t)
        t += 33

    assert result[0].x == pytest.approx(0.8, abs=0.01)


def test_reset_clears_state_so_next_value_passes_through():
    f = HandLandmarksFilter()
    f.apply(_make_landmarks(0.2, 0.2), timestamp_ms=0)
    f.apply(_make_landmarks(0.2, 0.2), timestamp_ms=33)

    f.reset()

    out = f.apply(_make_landmarks(0.9, 0.9), timestamp_ms=1000)
    assert out[0].x == pytest.approx(0.9)
    assert out[0].y == pytest.approx(0.9)


def test_z_is_passed_through_unfiltered():
    f = HandLandmarksFilter()
    f.apply(_make_landmarks(0.5, 0.5), timestamp_ms=0)
    landmarks = [Point(0.5, 0.5, z=0.42) for _ in range(NUM_LANDMARKS)]
    out = f.apply(landmarks, timestamp_ms=33)
    assert out[0].z == pytest.approx(0.42)


def test_apply_rejects_wrong_landmark_count():
    f = HandLandmarksFilter()
    with pytest.raises(ValueError):
        f.apply([Point(0.0, 0.0)] * 5, timestamp_ms=0)
