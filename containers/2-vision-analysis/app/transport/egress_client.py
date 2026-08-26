"""Container B → C 결과 패킷 전송 클라이언트.

Container C가 준비되기 전에도 B는 독립적으로 기동/재시도할 수 있어야 한다
(마이크로서비스 경계 — 2장 핵심 참조). 연결이 끊기면 지수 백오프로 재연결하며,
재연결 중 큐에 쌓인 패킷은 최신성을 위해 보존하지 않고 흘려보낸다(8.5절 원칙과 일관).
"""

from __future__ import annotations

import asyncio
import logging

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from ..config import Settings
from ..contracts import LandmarkPacket

logger = logging.getLogger(__name__)


async def run_egress_client(settings: Settings, out_queue: "asyncio.Queue[LandmarkPacket]") -> None:
    url = settings.transport.pattern_command_ws_url
    delay = settings.transport.egress_reconnect_min_delay
    max_delay = settings.transport.egress_reconnect_max_delay

    while True:
        try:
            async with connect(url) as ws:
                logger.info("egress connected to %s", url)
                delay = settings.transport.egress_reconnect_min_delay
                while True:
                    packet = await out_queue.get()
                    await ws.send(packet.to_json())
        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, OSError) as exc:
            logger.warning("egress connection to %s failed (%s); retrying in %.1fs", url, exc, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
