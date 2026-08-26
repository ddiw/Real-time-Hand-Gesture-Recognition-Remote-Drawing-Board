from app.metrics import MetricsCollector


def test_snapshot_reports_detection_rate():
    m = MetricsCollector(window_size=10)
    for present in [True, True, False, True]:
        m.record_inference(duration_ms=10.0, hand_present=present)

    snap = m.snapshot()
    assert snap.frames_total == 4
    assert snap.frames_hand_present == 3
    assert snap.detection_rate == 0.75


def test_snapshot_computes_percentiles():
    m = MetricsCollector(window_size=100)
    for ms in [10, 12, 11, 13, 100]:
        m.record_inference(duration_ms=ms, hand_present=True)

    snap = m.snapshot()
    assert snap.inference_ms_p50 < snap.inference_ms_p95


def test_large_spike_flagged_as_likely_palm_redetect():
    m = MetricsCollector(window_size=50)
    for _ in range(20):
        m.record_inference(duration_ms=10.0, hand_present=True)
    m.record_inference(duration_ms=30.0, hand_present=True)

    snap = m.snapshot()
    assert snap.palm_redetect_rate > 0


def test_empty_collector_snapshot_does_not_crash():
    m = MetricsCollector()
    snap = m.snapshot()
    assert snap.frames_total == 0
    assert snap.detection_rate == 0
