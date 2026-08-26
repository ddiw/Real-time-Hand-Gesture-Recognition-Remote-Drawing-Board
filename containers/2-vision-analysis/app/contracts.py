"""Container 간 인터페이스 계약.

- IngestFrameHeader : Container A → B, 프레임 메타데이터 (텍스트 프레임)
- LandmarkPacket    : Container B → C, PRD 6.1/6.2의 출력 스키마 및 계약 규칙
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .geometry import Point

VALID_PIXEL_FORMATS = ("rgb8", "bgr8")
VALID_ROTATIONS = (0, 90, 180, 270)


class ContractError(ValueError):
    """A↔B, B↔C 인터페이스 계약 위반."""


@dataclass(frozen=True)
class IngestFrameHeader:
    """Container A가 프레임마다 먼저 보내는 JSON 헤더.

    직후에 width*height*3 바이트의 binary 프레임(pixel 데이터)이 이어진다.
    """

    session_id: str
    seq: int
    capture_ts: int
    width: int
    height: int
    pixel_format: str
    rotation: int = 0
    mirrored: bool = False

    @classmethod
    def from_json(cls, raw: str | bytes) -> "IngestFrameHeader":
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ContractError(f"invalid ingest header JSON: {exc}") from exc

        try:
            header = cls(
                session_id=str(data["session_id"]),
                seq=int(data["seq"]),
                capture_ts=int(data["capture_ts"]),
                width=int(data["width"]),
                height=int(data["height"]),
                pixel_format=str(data["format"]),
                rotation=int(data.get("rotation", 0)),
                mirrored=bool(data.get("mirrored", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(f"malformed ingest header: {exc}") from exc

        if header.pixel_format not in VALID_PIXEL_FORMATS:
            raise ContractError(f"unsupported pixel format: {header.pixel_format}")
        if header.rotation not in VALID_ROTATIONS:
            raise ContractError(f"unsupported rotation: {header.rotation}")
        if header.width <= 0 or header.height <= 0:
            raise ContractError("width/height must be positive")

        return header

    @property
    def expected_payload_size(self) -> int:
        return self.width * self.height * 3


def _point_to_dict(p: Point) -> dict[str, float]:
    return {"x": p.x, "y": p.y, "z": p.z}


@dataclass(frozen=True)
class Handedness:
    label: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score}


@dataclass(frozen=True)
class Quality:
    near_edge: bool = False
    filtered: bool = False
    outlier_dropped: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "near_edge": self.near_edge,
            "filtered": self.filtered,
            "outlier_dropped": self.outlier_dropped,
        }


@dataclass(frozen=True)
class LandmarkPacket:
    """PRD 6.1 출력 스키마. present()/absent() 팩토리로만 생성해 계약 규칙 6.2-5를 강제한다."""

    session_id: str
    seq: int
    capture_ts: int
    processed_ts: int
    hand_present: bool
    frame_w: int
    frame_h: int
    handedness: Optional[Handedness] = None
    landmarks: Optional[Sequence[Point]] = None
    world_landmarks: Optional[Sequence[Point]] = None
    hand_scale: Optional[float] = None
    quality: Quality = field(default_factory=Quality)

    def __post_init__(self) -> None:
        if not self.hand_present:
            if self.landmarks is not None or self.world_landmarks is not None:
                raise ContractError(
                    "hand_present=false requires landmarks/world_landmarks to be None (rule 6.2-5)"
                )
        else:
            if self.landmarks is None or self.world_landmarks is None:
                raise ContractError("hand_present=true requires landmarks and world_landmarks")

    @classmethod
    def absent(
        cls,
        *,
        session_id: str,
        seq: int,
        capture_ts: int,
        processed_ts: int,
        frame_w: int,
        frame_h: int,
    ) -> "LandmarkPacket":
        return cls(
            session_id=session_id,
            seq=seq,
            capture_ts=capture_ts,
            processed_ts=processed_ts,
            hand_present=False,
            frame_w=frame_w,
            frame_h=frame_h,
        )

    @classmethod
    def present(
        cls,
        *,
        session_id: str,
        seq: int,
        capture_ts: int,
        processed_ts: int,
        frame_w: int,
        frame_h: int,
        handedness: Handedness,
        landmarks: Sequence[Point],
        world_landmarks: Sequence[Point],
        hand_scale: float,
        quality: Quality,
    ) -> "LandmarkPacket":
        return cls(
            session_id=session_id,
            seq=seq,
            capture_ts=capture_ts,
            processed_ts=processed_ts,
            hand_present=True,
            frame_w=frame_w,
            frame_h=frame_h,
            handedness=handedness,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            hand_scale=hand_scale,
            quality=quality,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "session_id": self.session_id,
            "seq": self.seq,
            "capture_ts": self.capture_ts,
            "processed_ts": self.processed_ts,
            "hand_present": self.hand_present,
            "frame": {"w": self.frame_w, "h": self.frame_h},
        }
        if self.hand_present:
            d["handedness"] = self.handedness.to_dict()
            d["landmarks"] = [_point_to_dict(p) for p in self.landmarks]
            d["world_landmarks"] = [_point_to_dict(p) for p in self.world_landmarks]
            d["hand_scale"] = self.hand_scale
            d["quality"] = self.quality.to_dict()
        else:
            d["handedness"] = None
            d["landmarks"] = None
            d["world_landmarks"] = None
            d["hand_scale"] = None
            d["quality"] = self.quality.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
