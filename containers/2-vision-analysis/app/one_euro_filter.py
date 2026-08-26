"""One Euro Filter — 실시간 랜드마크 스무딩 (PRD 8.2).

MediaPipe Hand Landmarker는 스무딩을 내장하지 않으므로(FR-B-13) 직접 구현한다.
참고: Casiez, Roussel, Vogel, "1-Euro Filter" (CHI 2012).
"""

from __future__ import annotations

import math
from typing import Sequence

from .geometry import NUM_LANDMARKS, Point


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1 - a) * x_prev


class _Axis1DFilter:
    """단일 스칼라 값(예: x 좌표 하나)에 대한 One Euro Filter."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, t: float, x: float) -> float:
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        t_e = max(t - self._t_prev, 1e-6)

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, x, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class HandLandmarksFilter:
    """21개 랜드마크의 x, y 각각에 독립적인 One Euro Filter를 적용한다.

    z는 필터링하지 않는다 (PRD 2.3 — z는 근사 상대 깊이, 형태 판별에는 world_landmarks 사용).
    손이 사라졌다 다시 나타나면 반드시 reset()을 호출해 이전 위치로 끌려오는 현상을 막는다.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff
        self._x_filters = [self._make_axis_filter() for _ in range(NUM_LANDMARKS)]
        self._y_filters = [self._make_axis_filter() for _ in range(NUM_LANDMARKS)]

    def _make_axis_filter(self) -> _Axis1DFilter:
        return _Axis1DFilter(self._min_cutoff, self._beta, self._d_cutoff)

    def reset(self) -> None:
        for f in self._x_filters:
            f.reset()
        for f in self._y_filters:
            f.reset()

    def apply(self, landmarks: Sequence[Point], timestamp_ms: float) -> list[Point]:
        if len(landmarks) != NUM_LANDMARKS:
            raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}")

        t = timestamp_ms / 1000.0
        out = []
        for i, lm in enumerate(landmarks):
            fx = self._x_filters[i].filter(t, lm.x)
            fy = self._y_filters[i].filter(t, lm.y)
            out.append(Point(fx, fy, lm.z))
        return out
