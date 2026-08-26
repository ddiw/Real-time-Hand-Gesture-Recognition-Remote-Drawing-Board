"""좌표 기하 연산: letterbox 매핑, hand_scale, 거리 계산.

PRD 2.3 (랜드마크 인덱스), 3장 (③ 리사이즈 / ⑦ 좌표 역매핑 / ⑧ hand_scale),
8.1 (거리 불변성) 참조.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21


class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class LetterboxParams:
    """원본 프레임을 목표 해상도에 종횡비 유지로 맞추기 위한 변환 정보."""

    scale: float
    pad_x: float
    pad_y: float
    src_w: int
    src_h: int
    dst_w: int
    dst_h: int


def compute_letterbox_params(src_w: int, src_h: int, dst_w: int, dst_h: int) -> LetterboxParams:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("width/height must be positive")
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    pad_x = (dst_w - new_w) / 2.0
    pad_y = (dst_h - new_h) / 2.0
    return LetterboxParams(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)


def letterbox_resize(image: np.ndarray, dst_w: int, dst_h: int) -> tuple[np.ndarray, LetterboxParams]:
    """종횡비를 유지한 채 (dst_w, dst_h) 캔버스 중앙에 리사이즈 배치한다 (FR-B-03)."""
    import cv2

    src_h, src_w = image.shape[:2]
    params = compute_letterbox_params(src_w, src_h, dst_w, dst_h)

    new_w = max(1, round(src_w * params.scale))
    new_h = max(1, round(src_h * params.scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    channels = image.shape[2] if image.ndim == 3 else 1
    canvas_shape = (dst_h, dst_w, channels) if image.ndim == 3 else (dst_h, dst_w)
    canvas = np.zeros(canvas_shape, dtype=image.dtype)

    off_x = round(params.pad_x)
    off_y = round(params.pad_y)
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

    return np.ascontiguousarray(canvas), params


def unletterbox_point(x: float, y: float, params: LetterboxParams) -> tuple[float, float]:
    """letterbox 캔버스 상의 정규화 좌표를 원본 프레임 기준 정규화 좌표로 역매핑한다 (FR-B-11).

    출력은 클램핑하지 않는다 (계약 규칙 6.2-2) — 프레임 밖 좌표는 음수/1 초과가 될 수 있다.
    """
    px = x * params.dst_w
    py = y * params.dst_h
    orig_x = (px - params.pad_x) / (params.scale * params.src_w)
    orig_y = (py - params.pad_y) / (params.scale * params.src_h)
    return orig_x, orig_y


def unletterbox_landmarks(landmarks: Sequence[Point], params: LetterboxParams) -> list[Point]:
    out = []
    for lm in landmarks:
        ox, oy = unletterbox_point(lm.x, lm.y, params)
        out.append(Point(ox, oy, lm.z))
    return out


def dist2d(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def dist3d(a: Point, b: Point) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def hand_scale(landmarks: Sequence[Point]) -> float:
    """손목~중지 MCP 거리. 화면상 손 크기의 기준값 (FR-B-12, PRD 12.3)."""
    return dist2d(landmarks[WRIST], landmarks[MIDDLE_MCP])


def is_near_edge(landmarks: Sequence[Point], margin: float) -> bool:
    """랜드마크 중 하나라도 프레임 경계 margin 이내면 True (FR-B-15)."""
    for lm in landmarks:
        if lm.x < margin or lm.x > 1.0 - margin or lm.y < margin or lm.y > 1.0 - margin:
            return True
    return False


def max_displacement(prev: Sequence[Point], curr: Sequence[Point]) -> float:
    """이전 프레임 대비 랜드마크별 최대 이동 거리 (정규화 좌표 기준, FR-B-14)."""
    return max(dist2d(p, c) for p, c in zip(prev, curr))
