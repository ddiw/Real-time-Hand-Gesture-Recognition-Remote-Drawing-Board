"""세션 단위 처리 파이프라인 (PRD 3장 전체, FR-B-04/09/11~16).

프레임 도착(동기, asyncio 루프 스레드) → 전처리 → MediaPipe 비동기 추론 제출.
추론 결과는 MediaPipe 내부 스레드의 콜백으로 도착하므로, 백그라운드 드레인 스레드가
이를 소비해 패킷으로 변환한 뒤 asyncio 루프로 안전하게 넘긴다.

동시에 하나의 프레임만 추론 중이도록 유지하고(latest-frame-wins, FR-B-04),
그 사이 도착한 프레임은 가장 최근 것만 남기고 버린다.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .contracts import Handedness, LandmarkPacket, Quality
from .geometry import LetterboxParams, hand_scale, is_near_edge, max_displacement, unletterbox_landmarks
from .landmarker import HandLandmarkerSession, LandmarkResult, make_mp_image
from .metrics import MetricsCollector
from .one_euro_filter import HandLandmarksFilter
from .preprocess import prepare_for_inference

logger = logging.getLogger(__name__)


@dataclass
class _RawFrame:
    """추론 대기 중인 미가공 프레임. 전처리 이전 상태로 보관한다."""

    seq: int
    capture_ts: int
    width: int
    height: int
    frame: np.ndarray
    pixel_format: str
    rotation: int
    mirrored: bool


@dataclass
class _PendingFrame:
    session_id: str
    seq: int
    capture_ts: int
    width: int
    height: int
    mp_image: object
    params: LetterboxParams


class SessionPipeline:
    """카메라 세션 하나에 대응하는 처리 파이프라인. 세션 시작 시 생성, 종료 시 close()."""

    def __init__(
        self,
        settings: Settings,
        session_id: str,
        metrics: MetricsCollector,
        loop: asyncio.AbstractEventLoop,
        out_queue: "asyncio.Queue[LandmarkPacket]",
    ):
        self._settings = settings
        self._session_id = session_id
        self._metrics = metrics
        self._loop = loop
        self._out_queue = out_queue

        self._result_queue: "queue.Queue[LandmarkResult]" = queue.Queue()
        self._landmarker = HandLandmarkerSession(settings.model, self._result_queue)
        self._filter = HandLandmarksFilter(
            min_cutoff=settings.one_euro.min_cutoff,
            beta=settings.one_euro.beta,
            d_cutoff=settings.one_euro.d_cutoff,
        )

        self._lock = threading.Lock()
        self._inflight = False
        self._pending_raw: _RawFrame | None = None
        self._context_by_ts: dict[int, _PendingFrame] = {}

        self._prev_landmarks = None
        self._hand_was_present = False

        self._closed = False
        self._drain_thread = threading.Thread(target=self._drain_loop, name=f"vision-drain-{session_id}", daemon=True)
        self._drain_thread.start()

    def offer_frame(
        self,
        seq: int,
        capture_ts: int,
        width: int,
        height: int,
        raw_frame: np.ndarray,
        pixel_format: str,
        rotation: int,
        mirrored: bool,
    ) -> None:
        """프레임 도착 시 호출된다. 처리 중이면 이전 대기 프레임을 버리고 최신 것으로 교체한다.

        전처리는 드롭 판정 이후로 미룬다. 이전에는 도착한 모든 프레임을 먼저
        전처리한 뒤 버릴지 결정했기 때문에, 버려질 프레임의 색변환·회전·letterbox
        비용을 asyncio 이벤트 루프 스레드에서 그대로 치렀다. 루프가 포화되면
        소켓 버퍼에 프레임이 쌓이고 그만큼 종단 지연이 늘어난다.
        """
        raw = _RawFrame(
            seq=seq,
            capture_ts=capture_ts,
            width=width,
            height=height,
            frame=raw_frame,
            pixel_format=pixel_format,
            rotation=rotation,
            mirrored=mirrored,
        )

        with self._lock:
            if self._inflight:
                self._pending_raw = raw
                return
            self._inflight = True
        # 락 밖에서 전처리한다. 락을 쥔 채로 하면 드레인 스레드가 전처리하는 동안
        # 이벤트 루프가 offer_frame에서 멈춘다.
        self._prepare_and_submit(raw)

    def _prepare_and_submit(self, raw: _RawFrame) -> None:
        prepared, params = prepare_for_inference(
            raw.frame,
            raw.pixel_format,
            raw.rotation,
            raw.mirrored,
            self._settings.pipeline.target_width,
            self._settings.pipeline.target_height,
            enable_clahe=self._settings.pipeline.enable_clahe,
        )
        pending = _PendingFrame(
            session_id=self._session_id,
            seq=raw.seq,
            capture_ts=raw.capture_ts,
            width=raw.width,
            height=raw.height,
            mp_image=make_mp_image(prepared),
            params=params,
        )
        ts = self._landmarker.submit(pending.mp_image, pending.capture_ts)
        self._context_by_ts[ts] = pending

    def _drain_loop(self) -> None:
        while not self._closed:
            try:
                result = self._result_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            packet = self._handle_result(result)

            with self._lock:
                next_raw = self._pending_raw
                self._pending_raw = None
                self._inflight = next_raw is not None
            # Preprocessing the queued frame here keeps that work off the
            # asyncio loop, which is free to keep draining the ingest socket.
            if next_raw is not None:
                self._prepare_and_submit(next_raw)

            if packet is not None:
                self._loop.call_soon_threadsafe(self._enqueue_packet, packet)

    def _enqueue_packet(self, packet: LandmarkPacket) -> None:
        try:
            self._out_queue.put_nowait(packet)
        except asyncio.QueueFull:
            logger.warning("session %s: egress queue full, dropping packet seq=%s", self._session_id, packet.seq)

    def _handle_result(self, result: LandmarkResult) -> LandmarkPacket | None:
        ctx = self._context_by_ts.pop(result.timestamp_ms, None)
        if ctx is None:
            logger.warning("session %s: no context for timestamp %s", self._session_id, result.timestamp_ms)
            return None

        self._metrics.record_inference(result.inference_ms, result.hand_present)
        processed_ts = int(time.time() * 1000)

        if not result.hand_present:
            if self._hand_was_present:
                self._filter.reset()
                self._prev_landmarks = None
            self._hand_was_present = False
            return LandmarkPacket.absent(
                session_id=ctx.session_id,
                seq=ctx.seq,
                capture_ts=ctx.capture_ts,
                processed_ts=processed_ts,
                frame_w=ctx.width,
                frame_h=ctx.height,
            )

        landmarks = unletterbox_landmarks(result.landmarks, ctx.params)
        filtered = self._filter.apply(landmarks, ctx.capture_ts)
        scale = hand_scale(filtered)
        near_edge = is_near_edge(filtered, self._settings.pipeline.near_edge_margin)

        outlier = False
        if self._prev_landmarks is not None and scale > 0:
            displacement = max_displacement(self._prev_landmarks, filtered)
            outlier = displacement > scale * self._settings.pipeline.outlier_scale_multiplier

        if not outlier:
            self._prev_landmarks = filtered
        self._hand_was_present = True

        return LandmarkPacket.present(
            session_id=ctx.session_id,
            seq=ctx.seq,
            capture_ts=ctx.capture_ts,
            processed_ts=processed_ts,
            frame_w=ctx.width,
            frame_h=ctx.height,
            handedness=Handedness(result.handedness_label, result.handedness_score),
            landmarks=filtered,
            world_landmarks=list(result.world_landmarks),
            hand_scale=scale,
            quality=Quality(near_edge=near_edge, filtered=True, outlier_dropped=outlier),
        )

    def close(self) -> None:
        self._closed = True
        self._drain_thread.join(timeout=2.0)
        self._landmarker.close()
