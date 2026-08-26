import numpy as np
import pytest

from app.geometry import (
    MIDDLE_MCP,
    WRIST,
    Point,
    compute_letterbox_params,
    hand_scale,
    is_near_edge,
    letterbox_resize,
    max_displacement,
    unletterbox_point,
)


def test_letterbox_params_wider_source_pads_vertically():
    params = compute_letterbox_params(src_w=1280, src_h=480, dst_w=640, dst_h=480)
    assert params.scale == pytest.approx(0.5)
    assert params.pad_x == pytest.approx(0.0)
    assert params.pad_y == pytest.approx((480 - 240) / 2)


def test_letterbox_params_taller_source_pads_horizontally():
    params = compute_letterbox_params(src_w=480, src_h=640, dst_w=640, dst_h=480)
    assert params.scale == pytest.approx(0.75)
    assert params.pad_y == pytest.approx(0.0)
    assert params.pad_x > 0


@pytest.mark.parametrize("src_w,src_h", [(1280, 480), (480, 640), (640, 480), (100, 100)])
def test_unletterbox_round_trip_recovers_known_point(src_w, src_h):
    dst_w, dst_h = 640, 480
    params = compute_letterbox_params(src_w, src_h, dst_w, dst_h)

    orig_x, orig_y = 0.3, 0.7
    px = (orig_x * src_w * params.scale + params.pad_x) / dst_w
    py = (orig_y * src_h * params.scale + params.pad_y) / dst_h

    recovered_x, recovered_y = unletterbox_point(px, py, params)
    assert recovered_x == pytest.approx(orig_x, abs=1e-6)
    assert recovered_y == pytest.approx(orig_y, abs=1e-6)


def test_unletterbox_does_not_clamp_out_of_frame_points():
    params = compute_letterbox_params(640, 480, 640, 480)
    x, y = unletterbox_point(-0.1, 1.2, params)
    assert x < 0
    assert y > 1.0


def test_letterbox_resize_preserves_aspect_and_pads():
    image = np.random.randint(0, 255, (480, 1280, 3), dtype=np.uint8)
    resized, params = letterbox_resize(image, 640, 480)
    assert resized.shape == (480, 640, 3)
    assert resized.flags["C_CONTIGUOUS"]
    assert params.pad_y > 0


def test_hand_scale_uses_wrist_and_middle_mcp():
    landmarks = [Point(0.0, 0.0)] * 21
    landmarks[WRIST] = Point(0.0, 0.0)
    landmarks[MIDDLE_MCP] = Point(0.3, 0.4)
    assert hand_scale(landmarks) == pytest.approx(0.5)


def test_is_near_edge_true_when_within_margin():
    landmarks = [Point(0.5, 0.5)] * 21
    landmarks[8] = Point(0.01, 0.5)
    assert is_near_edge(landmarks, margin=0.03) is True


def test_is_near_edge_false_when_comfortably_inside():
    landmarks = [Point(0.5, 0.5)] * 21
    assert is_near_edge(landmarks, margin=0.03) is False


def test_max_displacement_picks_largest_moved_landmark():
    prev = [Point(0.0, 0.0)] * 21
    curr = [Point(0.0, 0.0)] * 21
    curr[3] = Point(0.1, 0.0)
    curr[8] = Point(0.5, 0.0)
    assert max_displacement(prev, curr) == pytest.approx(0.5)
