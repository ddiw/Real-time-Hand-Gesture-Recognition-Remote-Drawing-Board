"""관측성: 추론 시간, 검출률, 팜 재검출 빈도 (FR-B-17, NFR-01/02/05/07/08).

MediaPipe Tasks API는 팜 검출이 실제로 실행됐는지를 결과 객체에 노출하지 않는다
(PRD 2.1 참조: ROI 트래킹은 완전히 내장되어 있다). 따라서 팜 재검출 여부는
추론 시간이 이동 중앙값 대비 큰 폭으로 튀는지를 보는 휴리스틱으로 근사한다
(PRD 4.1 보충: 재검출 프레임은 추론 시간이 2~3배로 튄다).
"""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Deque

import numpy as np

logger = logging.getLogger(__name__)

REDETECT_SPIKE_RATIO = 2.0


@dataclass(frozen=True)
class MetricsSnapshot:
    frames_total: int
    frames_hand_present: int
    detection_rate: float
    inference_ms_p50: float
    inference_ms_p95: float
    palm_redetect_rate: float

    def to_dict(self) -> dict:
        return {
            "frames_total": self.frames_total,
            "frames_hand_present": self.frames_hand_present,
            "detection_rate": self.detection_rate,
            "inference_ms_p50": self.inference_ms_p50,
            "inference_ms_p95": self.inference_ms_p95,
            "palm_redetect_rate": self.palm_redetect_rate,
        }


class MetricsCollector:
    def __init__(self, window_size: int = 300):
        self._window_size = window_size
        self._inference_ms: Deque[float] = deque(maxlen=window_size)
        self._hand_present: Deque[bool] = deque(maxlen=window_size)
        self._redetect_flags: Deque[bool] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._frames_total = 0
        self._frames_hand_present = 0

    def record_inference(self, duration_ms: float, hand_present: bool) -> None:
        with self._lock:
            median = float(np.median(self._inference_ms)) if self._inference_ms else duration_ms
            likely_redetect = median > 0 and duration_ms >= median * REDETECT_SPIKE_RATIO

            self._inference_ms.append(duration_ms)
            self._hand_present.append(hand_present)
            self._redetect_flags.append(likely_redetect)

            self._frames_total += 1
            if hand_present:
                self._frames_hand_present += 1

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            times = np.array(self._inference_ms) if self._inference_ms else np.array([0.0])
            window_n = len(self._hand_present) or 1
            detection_rate = sum(self._hand_present) / window_n
            redetect_rate = sum(self._redetect_flags) / window_n

            return MetricsSnapshot(
                frames_total=self._frames_total,
                frames_hand_present=self._frames_hand_present,
                detection_rate=detection_rate,
                inference_ms_p50=float(np.percentile(times, 50)),
                inference_ms_p95=float(np.percentile(times, 95)),
                palm_redetect_rate=redetect_rate,
            )


def _make_handler(collector: MetricsCollector) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
            logger.debug("metrics-http: " + format, *args)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._respond(200, {"status": "ok"})
            elif self.path == "/metrics":
                self._respond(200, collector.snapshot().to_dict())
            else:
                self._respond(404, {"error": "not found"})

        def _respond(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def run_metrics_http_server(collector: MetricsCollector, host: str, port: int) -> ThreadingHTTPServer:
    """/health, /metrics 를 서빙하는 백그라운드 HTTP 서버를 시작하고 서버 인스턴스를 반환한다."""
    server = ThreadingHTTPServer((host, port), _make_handler(collector))
    thread = threading.Thread(target=server.serve_forever, name="metrics-http", daemon=True)
    thread.start()
    logger.info("metrics http server listening on %s:%d", host, port)
    return server
