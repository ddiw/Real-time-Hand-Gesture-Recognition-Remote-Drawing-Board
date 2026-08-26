"""개발용 Container C 스탠드인.

Container C 없이 Container B(vision-analysis)만 단독으로 테스트할 때, B가 보내는
PRD 6.1 좌표 패킷을 수신해 콘솔에 사람이 읽을 수 있는 형태로 출력한다.

사용 예:
    python scripts/dev_pattern_sink.py
"""

from __future__ import annotations

import argparse
import asyncio
import json

from websockets.asyncio.server import ServerConnection, serve


def _format_packet(packet: dict) -> str:
    if not packet["hand_present"]:
        return f"seq={packet['seq']:>6}  hand not present"

    tip = packet["landmarks"][8]  # INDEX_TIP
    bar_width = 40
    pos = min(max(tip["x"], 0.0), 1.0)
    bar = "-" * int(pos * bar_width) + "o" + "-" * (bar_width - int(pos * bar_width))
    quality = packet["quality"]
    flags = ",".join(k for k, v in quality.items() if v) or "-"
    return (
        f"seq={packet['seq']:>6}  index_tip=({tip['x']:.3f}, {tip['y']:.3f})  "
        f"scale={packet['hand_scale']:.3f}  [{bar}]  quality={flags}"
    )


async def _handler(websocket: ServerConnection) -> None:
    print(f"Container C stand-in: session connected ({websocket.request.path})")
    async for message in websocket:
        packet = json.loads(message)
        print(_format_packet(packet))
    print("Container C stand-in: session disconnected")


async def run(host: str, port: int) -> None:
    async with serve(_handler, host, port) as server:
        print(f"listening on ws://{host}:{port}/landmarks (any path is accepted)")
        await server.wait_closed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Container C stand-in: print landmark packets from vision-analysis")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8761)
    args = parser.parse_args()
    asyncio.run(run(args.host, args.port))


if __name__ == "__main__":
    main()
