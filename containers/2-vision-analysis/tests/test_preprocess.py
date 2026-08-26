import numpy as np
import pytest

from app.preprocess import apply_geometric_correction, prepare_for_inference, to_rgb


def test_to_rgb_swaps_bgr_channels():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = [255, 0, 0]  # blue-dominant pixel in BGR

    rgb = to_rgb(bgr, "bgr8")
    assert rgb[0, 0].tolist() == [0, 0, 255]  # blue channel now last


def test_to_rgb_passthrough_for_rgb8():
    frame = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
    out = to_rgb(frame, "rgb8")
    assert np.array_equal(out, frame)


def test_to_rgb_rejects_unknown_format():
    with pytest.raises(ValueError):
        to_rgb(np.zeros((2, 2, 3), dtype=np.uint8), "yuv420")


def test_geometric_correction_rotation_90_changes_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rotated = apply_geometric_correction(frame, rotation=90, mirrored=False)
    assert rotated.shape[:2] == (640, 480)


def test_geometric_correction_mirror_flips_horizontally():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[0, 0] = [9, 9, 9]
    mirrored = apply_geometric_correction(frame, rotation=0, mirrored=True)
    assert mirrored[0, -1].tolist() == [9, 9, 9]


def test_geometric_correction_rejects_invalid_rotation():
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_geometric_correction(frame, rotation=45, mirrored=False)


def test_prepare_for_inference_produces_contiguous_target_shape():
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    prepared, params = prepare_for_inference(
        frame, "bgr8", rotation=0, mirrored=False, target_w=640, target_h=480
    )
    assert prepared.shape == (480, 640, 3)
    assert prepared.dtype == np.uint8
    assert prepared.flags["C_CONTIGUOUS"]
    assert params.dst_w == 640 and params.dst_h == 480
