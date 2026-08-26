"""MediaPipe Hand Landmarker 래퍼 (PRD 2.4, 4장, 12.1 부록 참조).

`LIVE_STREAM` 모드는 백그라운드 C++ 스레드에서 결과 콜백을 호출한다. 이 모듈은
그 콜백을 스레드-세이프 큐에 적재해 asyncio 이벤트 루프에서 안전하게 소비할 수
있도록 다리를 놓는다 (FR-B-06, FR-B-07, FR-B-09).
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .config import ModelConfig
from .geometry import Point

logger = logging.getLogger(__name__)

_DELEGATE_MAP = {
    "CPU": mp_python.BaseOptions.Delegate.CPU,
    "GPU": mp_python.BaseOptions.Delegate.GPU,
}


@dataclass
class LandmarkResult:
    timestamp_ms: int
    hand_present: bool
    landmarks: Optional[list[Point]] = None
    world_landmarks: Optional[list[Point]] = None
    handedness_label: Optional[str] = None
    handedness_score: Optional[float] = None
    inference_ms: float = 0.0


class MonotonicTimestampGuard:
    """detect_async에 전달하는 타임스탬프의 단조증가를 보장한다 (FR-B-07, 8.3 함정).

    위반 시 MediaPipe가 예외를 던지고 프로세스가 중단되므로, 여기서 선제적으로 보정한다.
    """

    def __init__(self) -> None:
        self._last_ts = -1

    def next(self, capture_ts_ms: int) -> int:
        ts = capture_ts_ms if capture_ts_ms > self._last_ts else self._last_ts + 1
        self._last_ts = ts
        return ts


class HandLandmarkerSession:
    """세션(연결) 단위로 하나씩 생성/해제하는 HandLandmarker 인스턴스 (FR-B-09).

    인스턴스를 세션 간 공유하면 내부 트래킹 상태가 오염된다(8.3 함정) — 반드시
    세션마다 새로 만들고 close()로 해제할 것.
    """

    def __init__(self, model: ModelConfig, result_queue: "queue.Queue[LandmarkResult]"):
        self._model_cfg = model
        self._result_queue = result_queue
        self._ts_guard = MonotonicTimestampGuard()
        self._pending_started_at: dict[int, float] = {}

        delegate = _DELEGATE_MAP.get(model.delegate.upper(), mp_python.BaseOptions.Delegate.CPU)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model.asset_path,
                delegate=delegate,
            ),
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=model.num_hands,
            min_hand_detection_confidence=model.min_hand_detection_confidence,
            min_hand_presence_confidence=model.min_hand_presence_confidence,
            min_tracking_confidence=model.min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def submit(self, mp_image: "mp.Image", capture_ts_ms: int) -> int:
        """프레임을 비동기 추론에 투입한다. 실제 사용된 타임스탬프를 반환한다 (FR-B-06)."""
        ts = self._ts_guard.next(capture_ts_ms)
        self._pending_started_at[ts] = time.monotonic()
        self._landmarker.detect_async(mp_image, ts)
        return ts

    def _on_result(self, result, output_image, timestamp_ms: int) -> None:
        started_at = self._pending_started_at.pop(timestamp_ms, None)
        inference_ms = (time.monotonic() - started_at) * 1000.0 if started_at else 0.0

        if not result.hand_landmarks:
            self._result_queue.put(
                LandmarkResult(timestamp_ms=timestamp_ms, hand_present=False, inference_ms=inference_ms)
            )
            return

        lms = [Point(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        world = [Point(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]]
        handed = result.handedness[0][0]

        self._result_queue.put(
            LandmarkResult(
                timestamp_ms=timestamp_ms,
                hand_present=True,
                landmarks=lms,
                world_landmarks=world,
                handedness_label=handed.category_name,
                handedness_score=handed.score,
                inference_ms=inference_ms,
            )
        )

    def close(self) -> None:
        self._landmarker.close()


def make_mp_image(rgb_frame) -> "mp.Image":
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
