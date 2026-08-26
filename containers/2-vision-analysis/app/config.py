"""Container B (vision-analysis) 런타임 설정.

모든 값은 환경변수로 오버라이드 가능하며, 기본값은 docker-compose 서비스명을
기준으로 한다 (PRD 4장 파라미터, 6장 인터페이스 계약 참조).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class ModelConfig:
    """PRD 4장 — Hand Landmarker 설정 파라미터."""

    asset_path: str = field(default_factory=lambda: _env_str(
        "HAND_LANDMARKER_MODEL_PATH", "models/hand_landmarker.task"
    ))
    num_hands: int = field(default_factory=lambda: _env_int("VISION_NUM_HANDS", 1))
    min_hand_detection_confidence: float = field(
        default_factory=lambda: _env_float("VISION_MIN_DETECTION_CONFIDENCE", 0.7)
    )
    min_hand_presence_confidence: float = field(
        default_factory=lambda: _env_float("VISION_MIN_PRESENCE_CONFIDENCE", 0.6)
    )
    min_tracking_confidence: float = field(
        default_factory=lambda: _env_float("VISION_MIN_TRACKING_CONFIDENCE", 0.5)
    )
    delegate: str = field(default_factory=lambda: _env_str("VISION_DELEGATE", "CPU"))


@dataclass(frozen=True)
class PipelineConfig:
    """PRD 3장 — 전처리/후처리 파이프라인 설정."""

    target_width: int = field(default_factory=lambda: _env_int("VISION_TARGET_WIDTH", 640))
    target_height: int = field(default_factory=lambda: _env_int("VISION_TARGET_HEIGHT", 480))
    target_fps: float = field(default_factory=lambda: _env_float("VISION_TARGET_FPS", 30.0))
    enable_clahe: bool = field(default_factory=lambda: _env_bool("VISION_ENABLE_CLAHE", False))
    outlier_scale_multiplier: float = field(
        default_factory=lambda: _env_float("VISION_OUTLIER_SCALE_MULTIPLIER", 4.0)
    )
    near_edge_margin: float = field(
        default_factory=lambda: _env_float("VISION_NEAR_EDGE_MARGIN", 0.03)
    )


@dataclass(frozen=True)
class OneEuroConfig:
    """PRD 8.2 — One Euro Filter 튜닝 파라미터."""

    min_cutoff: float = field(default_factory=lambda: _env_float("VISION_EURO_MIN_CUTOFF", 1.0))
    beta: float = field(default_factory=lambda: _env_float("VISION_EURO_BETA", 0.3))
    d_cutoff: float = field(default_factory=lambda: _env_float("VISION_EURO_D_CUTOFF", 1.0))


@dataclass(frozen=True)
class TransportConfig:
    """Container A(ingest)/C(egress) 연결 설정.

    Container A는 이 서버로 프레임을 스트리밍하고(WebSocket 클라이언트),
    Container B는 결과 패킷을 Container C로 스트리밍한다(WebSocket 클라이언트).
    """

    ingest_host: str = field(default_factory=lambda: _env_str("VISION_INGEST_HOST", "0.0.0.0"))
    ingest_port: int = field(default_factory=lambda: _env_int("VISION_INGEST_PORT", 8760))

    pattern_command_ws_url: str = field(
        default_factory=lambda: _env_str(
            "PATTERN_COMMAND_WS_URL", "ws://pattern-command:8761/landmarks"
        )
    )
    egress_reconnect_min_delay: float = field(
        default_factory=lambda: _env_float("VISION_EGRESS_RECONNECT_MIN_DELAY", 0.5)
    )
    egress_reconnect_max_delay: float = field(
        default_factory=lambda: _env_float("VISION_EGRESS_RECONNECT_MAX_DELAY", 10.0)
    )

    metrics_host: str = field(default_factory=lambda: _env_str("VISION_METRICS_HOST", "0.0.0.0"))
    metrics_port: int = field(default_factory=lambda: _env_int("VISION_METRICS_PORT", 8762))


@dataclass(frozen=True)
class Settings:
    model: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    one_euro: OneEuroConfig = field(default_factory=OneEuroConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    log_level: str = field(default_factory=lambda: _env_str("VISION_LOG_LEVEL", "INFO"))


def load_settings() -> Settings:
    return Settings()
