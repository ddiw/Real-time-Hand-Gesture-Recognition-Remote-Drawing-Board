from app.landmarker import MonotonicTimestampGuard


def test_timestamp_guard_passes_through_increasing_values():
    guard = MonotonicTimestampGuard()
    assert guard.next(100) == 100
    assert guard.next(200) == 200


def test_timestamp_guard_bumps_non_increasing_values():
    guard = MonotonicTimestampGuard()
    assert guard.next(100) == 100
    assert guard.next(100) == 101
    assert guard.next(50) == 102


def test_timestamp_guard_starts_from_first_value():
    guard = MonotonicTimestampGuard()
    assert guard.next(0) == 0
    assert guard.next(0) == 1
