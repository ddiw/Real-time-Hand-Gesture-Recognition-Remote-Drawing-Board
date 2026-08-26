"""프레임 전처리: 색공간 정규화, 기하 보정, letterbox 리사이즈 (PRD 3장 ①~④, FR-B-01~05).

MediaPipe는 BGR을 지원하지 않으므로(8.3 함정) 반드시 RGB로 변환한 뒤 사용해야 한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import LetterboxParams, letterbox_resize

_ROTATION_TO_CV2 = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def to_rgb(frame: np.ndarray, pixel_format: str) -> np.ndarray:
    """입력 프레임을 RGB uint8 배열로 변환한다 (FR-B-01)."""
    if pixel_format == "rgb8":
        rgb = frame
    elif pixel_format == "bgr8":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"unsupported pixel format: {pixel_format}")

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return rgb


def apply_geometric_correction(frame: np.ndarray, rotation: int, mirrored: bool) -> np.ndarray:
    """rotation/mirrored 플래그를 추론 이전에 보정한다 (FR-B-02).

    Container A가 이미 실제 카메라 방향에 맞춰 rotation을 계산해 전달한다고 가정하고,
    여기서는 그 값을 그대로 적용한다. mirrored=True는 전면 카메라 좌우반전을 되돌린다.
    """
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f"unsupported rotation: {rotation}")

    out = frame
    if rotation != 0:
        out = cv2.rotate(out, _ROTATION_TO_CV2[rotation])
    if mirrored:
        out = cv2.flip(out, 1)
    return out


def apply_clahe(frame_rgb: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """저조도 프레임에 CLAHE 기반 대비 보정을 적용한다 (FR-B-05)."""
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l_channel)

    merged = cv2.merge((l_eq, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def prepare_for_inference(
    frame: np.ndarray,
    pixel_format: str,
    rotation: int,
    mirrored: bool,
    target_w: int,
    target_h: int,
    enable_clahe: bool = False,
) -> tuple[np.ndarray, LetterboxParams]:
    """PRD 3장 ①~④ 단계를 순서대로 적용해 MediaPipe 입력용 배열을 만든다.

    반환된 배열은 RGB, uint8, C-contiguous이며 (target_h, target_w, 3) 형상을 가진다.
    """
    rgb = to_rgb(frame, pixel_format)
    rgb = apply_geometric_correction(rgb, rotation, mirrored)
    if enable_clahe:
        rgb = apply_clahe(rgb)
    resized, params = letterbox_resize(rgb, target_w, target_h)
    return np.ascontiguousarray(resized), params
