"""Container B (영상 분석 엔진) 진입점.

- ingest server : Container A → B 프레임 수신 (ws://.../ingest/{session_id})
- egress client : B → Container C 좌표 패킷 전송
- metrics server: /health, /metrics (FR-B-17)
"""

from __future__ import annotations

import asyncio
import logging
import signal

from .config import Settings, load_settings
from .contracts import LandmarkPacket
from .metrics import MetricsCollector, run_metrics_http_server
from .transport.egress_client import run_egress_client
from .transport.ingest_server import run_ingest_server

logger = logging.getLogger(__name__)

EGRESS_QUEUE_MAXSIZE = 8


async def run(settings: Settings) -> None:
    metrics = MetricsCollector()
    run_metrics_http_server(metrics, settings.transport.metrics_host, settings.transport.metrics_port)

    out_queue: "asyncio.Queue[LandmarkPacket]" = asyncio.Queue(maxsize=EGRESS_QUEUE_MAXSIZE)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_ingest_server(settings, metrics, out_queue), name="ingest-server")
        tg.create_task(run_egress_client(settings, out_queue), name="egress-client")

        loop = asyncio.get_running_loop()
        stop = loop.create_future()

        def _request_shutdown() -> None:
            if not stop.done():
                logger.info("shutdown signal received")
                stop.set_result(None)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _request_shutdown)
            except NotImplementedError:
                pass  # Windows에서 SIGTERM 핸들러 등록 불가 — Ctrl+C(SIGINT)만 지원

        await stop
        raise SystemExit(0)


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run(settings))
    except (SystemExit, KeyboardInterrupt):
        logger.info("vision-analysis shutting down")


if __name__ == "__main__":
    main()
