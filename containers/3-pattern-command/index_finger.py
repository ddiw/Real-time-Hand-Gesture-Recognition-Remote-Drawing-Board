from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import acos, degrees, dist
from typing import Protocol, Sequence


class Landmark(Protocol):
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class IndexFingerState:
    raw_label: str
    stable_label: str
    pip_angle_deg: float
    dip_angle_deg: float
    straightness_ratio: float
    open_votes: int
    window_size: int


def _vector(origin: Landmark, target: Landmark) -> tuple[float, float, float]:
    return (target.x - origin.x, target.y - origin.y, target.z - origin.z)


def joint_angle(a: Landmark, vertex: Landmark, c: Landmark) -> float:
    first = _vector(vertex, a)
    second = _vector(vertex, c)
    first_length = sum(value * value for value in first) ** 0.5
    second_length = sum(value * value for value in second) ** 0.5
    if first_length == 0.0 or second_length == 0.0:
        return 0.0
    cosine = sum(x * y for x, y in zip(first, second)) / (first_length * second_length)
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def distance_3d(a: Landmark, b: Landmark) -> float:
    return dist((a.x, a.y, a.z), (b.x, b.y, b.z))


class IndexFingerClassifier:
    """Classifies a finger from four 3D MediaPipe world-landmark indices."""

    def __init__(
        self,
        open_pip_angle_deg: float = 120.0,
        window_size: int = 5,
        required_open_votes: int = 4,
        finger_indices: tuple[int, int, int, int] = (5, 6, 7, 8),
    ) -> None:
        if not 1 <= required_open_votes <= window_size:
            raise ValueError("required_open_votes must be within window_size")
        self.open_pip_angle_deg = open_pip_angle_deg
        self.window_size = window_size
        self.required_open_votes = required_open_votes
        self.mcp_index, self.pip_index, self.dip_index, self.tip_index = finger_indices
        self._history: deque[bool] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    def update(self, landmarks: Sequence[Landmark]) -> IndexFingerState:
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")

        pip_angle = joint_angle(
            landmarks[self.mcp_index], landmarks[self.pip_index], landmarks[self.dip_index]
        )
        dip_angle = joint_angle(
            landmarks[self.pip_index], landmarks[self.dip_index], landmarks[self.tip_index]
        )
        joint_path_length = (
            distance_3d(landmarks[self.mcp_index], landmarks[self.pip_index])
            + distance_3d(landmarks[self.pip_index], landmarks[self.dip_index])
            + distance_3d(landmarks[self.dip_index], landmarks[self.tip_index])
        )
        straightness_ratio = (
            distance_3d(landmarks[self.mcp_index], landmarks[self.tip_index]) / joint_path_length
            if joint_path_length > 0.0
            else 0.0
        )
        # The captured depth-direction test data separates OPEN and CLOSED
        # cleanly at the PIP joint. DIP and straightness remain diagnostic
        # values because perspective can make them unreliable constraints.
        is_open = pip_angle >= self.open_pip_angle_deg
        self._history.append(is_open)

        open_votes = sum(self._history)
        enough_frames = len(self._history) == self.window_size
        stable_open = enough_frames and open_votes >= self.required_open_votes
        stable_label = "OPEN" if stable_open else "CLOSED"
        return IndexFingerState(
            raw_label="OPEN" if is_open else "CLOSED",
            stable_label=stable_label,
            pip_angle_deg=pip_angle,
            dip_angle_deg=dip_angle,
            straightness_ratio=straightness_ratio,
            open_votes=open_votes,
            window_size=self.window_size,
        )
