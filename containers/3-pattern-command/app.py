"""Container C WebSocket server: PRD 6.1 landmarks to canvas commands."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from gesture_classifier import GestureClassifier


HOST = os.getenv("PATTERN_COMMAND_HOST", "0.0.0.0")
PORT = int(os.getenv("PATTERN_COMMAND_PORT", "8761"))
CANVAS_WS_URL = os.getenv("CANVAS_WS_URL", "ws://127.0.0.1:8770/commands/{session_id}")
# How far, in palm widths, the hand must sit from the neutral point before the
# three-finger posture starts zooming. Raise it to widen the dead zone.
ZOOM_DEADZONE_RATIO = float(os.getenv("ZOOM_DEADZONE_RATIO", "0.15"))
ZOOM_FILTER_ALPHA = float(os.getenv("ZOOM_FILTER_ALPHA", "0.6"))
# Frames a hand must be tracked before a fist counts, and frames the fist must
# be held before it clears. At 15 FPS each is roughly 0.87 seconds.
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", "13"))
CLEAR_HOLD_FRAMES = int(os.getenv("CLEAR_HOLD_FRAMES", "13"))
FINGER_WINDOW = int(os.getenv("FINGER_WINDOW", "5"))
FINGER_OPEN_VOTES = int(os.getenv("FINGER_OPEN_VOTES", "4"))
logging.basicConfig(level=os.getenv("PATTERN_COMMAND_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0


class CommandProcessor:
    def __init__(self) -> None:
        self.classifiers: dict[str, GestureClassifier] = {}

    def process(self, packet: dict) -> dict:
        session_id = str(packet["session_id"])
        classifier = self.classifiers.setdefault(
            session_id,
            GestureClassifier(
                zoom_deadzone_ratio=ZOOM_DEADZONE_RATIO,
                zoom_filter_alpha=ZOOM_FILTER_ALPHA,
                warmup_frames=WARMUP_FRAMES,
                clear_hold_frames=CLEAR_HOLD_FRAMES,
                finger_window=FINGER_WINDOW,
                finger_open_votes=FINGER_OPEN_VOTES,
            ),
        )
        landmarks = packet.get("landmarks")
        world = packet.get("world_landmarks")
        if not packet.get("hand_present") or not landmarks or not world:
            classifier.reset()
            return self._base(packet, command="IDLE", mode="IDLE")
        if len(landmarks) != 21 or len(world) != 21:
            raise ValueError("expected exactly 21 landmarks and world_landmarks")

        screen = [Point(**item) for item in landmarks]
        state = classifier.update([Point(**item) for item in world], screen)
        tip, pip = landmarks[8], landmarks[6]
        result = self._base(packet, command=state.command, mode=state.mode)
        result.update({
            "index_tip": {"x": float(tip["x"]), "y": float(tip["y"])},
            "index_direction": {
                "x": float(tip["x"]) - float(pip["x"]),
                "y": float(tip["y"]) - float(pip["y"]),
            },
            "zoom_direction": state.zoom_direction,
            "finger_count": state.finger_count,
            # The monitor draws this skeleton over the phone's source frame.
            # Four decimals is well under one pixel at 640x480.
            "landmarks": [
                {"x": round(point.x, 4), "y": round(point.y, 4)} for point in screen
            ],
        })
        return result

    @staticmethod
    def _base(packet: dict, *, command: str, mode: str) -> dict:
        capture_ts = int(packet.get("capture_ts", 0))
        processed_ts = int(packet.get("processed_ts", capture_ts))
        return {
            "schema_version": "1.0",
            "session_id": str(packet["session_id"]),
            "frame_id": packet.get("frame_id"),
            "seq": int(packet.get("seq", 0)),
            "command": command,
            "mode": mode,
            "inference_ms": max(0, processed_ts - capture_ts),
        }


processor = CommandProcessor()
canvas_connections = {}


async def forward_to_canvas(session_id: str, command: dict) -> None:
    url = CANVAS_WS_URL.format(session_id=session_id)
    delay = 0.25
    while True:
        try:
            canvas = canvas_connections.get(session_id)
            if canvas is None:
                canvas = await connect(url)
                canvas_connections[session_id] = canvas
            await canvas.send(json.dumps(command, separators=(",", ":")))
            return
        except (ConnectionClosed, OSError):
            stale = canvas_connections.pop(session_id, None)
            if stale is not None:
                await stale.close()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)


async def landmarks_handler(websocket) -> None:
    if websocket.request.path.split("?", 1)[0] != "/landmarks":
        await websocket.close(code=1008, reason="expected /landmarks")
        return
    try:
        async for message in websocket:
            if not isinstance(message, str):
                await websocket.close(code=1003, reason="landmark packet must be JSON text")
                return
            try:
                command = processor.process(json.loads(message))
                await forward_to_canvas(command["session_id"], command)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("invalid landmark packet: %s", exc)
                await websocket.send(json.dumps({"error": str(exc)}))
    except ConnectionClosed:
        pass


async def main() -> None:
    async with serve(landmarks_handler, HOST, PORT, max_size=2 * 1024 * 1024):
        logger.info("pattern command listening on ws://%s:%d/landmarks", HOST, PORT)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
