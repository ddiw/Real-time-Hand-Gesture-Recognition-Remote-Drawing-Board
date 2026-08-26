"""개발용 Container A 스탠드인.

Container A 없이 Container B(vision-analysis)만 단독으로 켜서 테스트할 때,
웹캠 또는 로컬 영상 파일을 ingest 프로토콜(JSON 헤더 + binary 프레임)에 맞춰
스트리밍한다. PRD 6.2 계약을 실제로 준수하는 최소 참조 클라이언트이기도 하다.

사용 예:
    python scripts/dev_camera_source.py --source 0          # 웹캠
    python scripts/dev_camera_source.py --source clip.mp4   # 영상 파일
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import cv2
from websockets.asyncio.client import connect


async def stream(url: str, source: int | str, target_fps: float, rotation: int, mirrored: bool) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"failed to open video source: {source}")

    frame_interval = 1.0 / target_fps
    seq = 0

    async with connect(url, max_size=None) as ws:
        print(f"connected to {url}")
        try:
            while True:
                loop_start = time.monotonic()
                ok, frame = cap.read()
                if not ok:
                    print("end of stream (source exhausted or disconnected)")
                    break

                height, width = frame.shape[:2]
                header = {
                    "session_id": url.rsplit("/", 1)[-1],
                    "seq": seq,
                    "capture_ts": int(time.time() * 1000),
                    "width": width,
                    "height": height,
                    "format": "bgr8",  # cv2 기본 색공간 — Container B가 RGB로 변환한다 (FR-B-01)
                    "rotation": rotation,
                    "mirrored": mirrored,
                }
                await ws.send(json.dumps(header))
                await ws.send(frame.tobytes())

                if seq % 30 == 0:
                    print(f"sent seq={seq} ({width}x{height})")
                seq += 1

                elapsed = time.monotonic() - loop_start
                await asyncio.sleep(max(0.0, frame_interval - elapsed))
        finally:
            cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Container A stand-in: stream a camera/video to vision-analysis")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--session-id", default="dev-session")
    parser.add_argument("--source", default="0", help="webcam index (0, 1, ...) or path to a video file")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rotation", type=int, default=0, choices=[0, 90, 180, 270])
    parser.add_argument("--mirror", action="store_true", help="undo front-camera mirroring before sending")
    args = parser.parse_args()

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    url = f"ws://{args.host}:{args.port}/ingest/{args.session_id}"
    asyncio.run(stream(url, source, args.fps, args.rotation, args.mirror))


if __name__ == "__main__":
    main()
