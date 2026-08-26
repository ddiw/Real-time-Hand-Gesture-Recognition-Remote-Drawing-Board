from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from index_finger import IndexFingerClassifier, IndexFingerState, Landmark, distance_3d


@dataclass(frozen=True)
class GestureState:
    command: str
    mode: str
    release_frames: int
    index: IndexFingerState
    middle: IndexFingerState
    ring: IndexFingerState
    pinky: IndexFingerState
    thumb_active: bool
    zoom_mode_active: bool
    thumb_index_spacing_ratio: float | None
    zoom_session_direction: str | None
    zoom_delta: float | None


class GestureClassifier:
    """Container C rule engine for drawing, erasing, and thumb-index zoom."""

    def __init__(
        self,
        open_pip_angle_deg: float = 120.0,
        thumb_active_ratio: float = 0.65,
        zoom_start_closed_ratio: float = 0.80,
        zoom_start_open_ratio: float = 1.00,
        zoom_start_confirm_frames: int = 3,
        zoom_motion_ratio: float = 0.05,
        zoom_filter_alpha: float = 1.0,
        release_confirm_frames: int = 3,
        release_pip_angle_deg: float | None = None,
    ) -> None:
        if not 0.0 < zoom_filter_alpha <= 1.0:
            raise ValueError("zoom_filter_alpha must be within (0, 1]")
        self.index_classifier = IndexFingerClassifier(
            open_pip_angle_deg=open_pip_angle_deg,
            finger_indices=(5, 6, 7, 8),
        )
        self.middle_classifier = IndexFingerClassifier(
            open_pip_angle_deg=open_pip_angle_deg,
            finger_indices=(9, 10, 11, 12),
        )
        self.ring_classifier = IndexFingerClassifier(
            open_pip_angle_deg=open_pip_angle_deg,
            finger_indices=(13, 14, 15, 16),
        )
        self.pinky_classifier = IndexFingerClassifier(
            open_pip_angle_deg=open_pip_angle_deg,
            finger_indices=(17, 18, 19, 20),
        )
        self.thumb_active_ratio = thumb_active_ratio
        self.zoom_start_closed_ratio = zoom_start_closed_ratio
        self.zoom_start_open_ratio = zoom_start_open_ratio
        self.zoom_start_confirm_frames = zoom_start_confirm_frames
        self.zoom_motion_ratio = zoom_motion_ratio
        self.zoom_filter_alpha = zoom_filter_alpha
        self.release_confirm_frames = release_confirm_frames
        self.release_pip_angle_deg = (
            open_pip_angle_deg if release_pip_angle_deg is None else release_pip_angle_deg
        )
        self._active_mode = "IDLE"
        self._release_frames = 0
        self._zoom_mode_active = False
        self._zoom_start_frames = 0
        self._zoom_start_ratio: float | None = None
        self._zoom_session_direction: str | None = None
        self._filtered_zoom_ratio: float | None = None

    def reset(self) -> None:
        self.index_classifier.reset()
        self.middle_classifier.reset()
        self.ring_classifier.reset()
        self.pinky_classifier.reset()
        self._active_mode = "IDLE"
        self._release_frames = 0
        self._zoom_mode_active = False
        self._zoom_start_frames = 0
        self._zoom_start_ratio = None
        self._zoom_session_direction = None
        self._filtered_zoom_ratio = None

    def update(self, landmarks: Sequence[Landmark]) -> GestureState:
        index = self.index_classifier.update(landmarks)
        middle = self.middle_classifier.update(landmarks)
        ring = self.ring_classifier.update(landmarks)
        pinky = self.pinky_classifier.update(landmarks)
        index_only_open = (
            index.stable_label == "OPEN"
            and middle.stable_label == "CLOSED"
            and ring.stable_label == "CLOSED"
            and pinky.stable_label == "CLOSED"
        )
        two_fingers_open = (
            index.stable_label == "OPEN"
            and middle.stable_label == "OPEN"
            and ring.stable_label == "CLOSED"
            and pinky.stable_label == "CLOSED"
        )
        raw_index_only_open = (
            index.raw_label == "OPEN"
            and middle.raw_label == "CLOSED"
            and ring.raw_label == "CLOSED"
            and pinky.raw_label == "CLOSED"
        )

        spacing_ratio = None
        zoom_delta = None
        palm_width = distance_3d(landmarks[5], landmarks[17])
        thumb_extension_ratio = (
            distance_3d(landmarks[4], landmarks[5]) / palm_width
            if palm_width > 0.0
            else 0.0
        )
        thumb_active = index_only_open and thumb_extension_ratio >= self.thumb_active_ratio
        raw_zoom_candidate = (
            raw_index_only_open and thumb_extension_ratio >= self.thumb_active_ratio
        )

        allow_zoom_evaluation = self._active_mode in ("IDLE", "ZOOM")
        if allow_zoom_evaluation and (index_only_open or raw_zoom_candidate):
            # Once zoom starts, retain its control mode for the full index-only
            # gesture. Thumb tracking can momentarily fluctuate during a pinch;
            # it must not send the command back to DRAW in that interval.
            spacing_ratio = distance_3d(landmarks[4], landmarks[8]) / palm_width if palm_width > 0.0 else 0.0
            if not self._zoom_mode_active and raw_zoom_candidate:
                self._zoom_start_frames += 1
                # Count the three confirmation frames while the general finger
                # stabilizer is warming up, but never lock before the stable
                # index-only posture itself has been confirmed.
                if (
                    self._zoom_start_frames >= self.zoom_start_confirm_frames
                    and index_only_open
                ):
                    if spacing_ratio <= self.zoom_start_closed_ratio:
                        self._zoom_session_direction = "ZOOM_IN"
                    elif spacing_ratio >= self.zoom_start_open_ratio:
                        self._zoom_session_direction = "ZOOM_OUT"
                    if self._zoom_session_direction is not None:
                        self._zoom_mode_active = True
                        self._zoom_start_ratio = spacing_ratio
                        self._filtered_zoom_ratio = spacing_ratio
        elif self._active_mode == "IDLE":
            self._zoom_start_frames = 0

        if self._active_mode != "IDLE":
            if index.pip_angle_deg < self.release_pip_angle_deg:
                self._release_frames += 1
            else:
                self._release_frames = 0

            if self._release_frames >= self.release_confirm_frames:
                self._active_mode = "IDLE"
                self._release_frames = 0
                self._zoom_mode_active = False
                self._zoom_start_frames = 0
                self._zoom_start_ratio = None
                self._zoom_session_direction = None
                self._filtered_zoom_ratio = None
                command = "IDLE"
            elif self._active_mode == "DRAW":
                command = "DRAW"
            elif self._active_mode == "ERASE":
                command = "ERASE"
            else:
                command = self._zoom_command_from_motion(spacing_ratio)
        else:
            if self._zoom_mode_active:
                self._active_mode = "ZOOM"
                command = self._zoom_command_from_motion(spacing_ratio)
            elif index_only_open and not thumb_active:
                self._active_mode = "DRAW"
                command = "DRAW"
            elif two_fingers_open:
                self._active_mode = "ERASE"
                command = "ERASE"
            else:
                command = "IDLE"

        return GestureState(
            command=command,
            mode=self._active_mode,
            release_frames=self._release_frames,
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            thumb_active=thumb_active,
            zoom_mode_active=self._zoom_mode_active,
            thumb_index_spacing_ratio=spacing_ratio,
            zoom_session_direction=self._zoom_session_direction,
            zoom_delta=(spacing_ratio - self._zoom_start_ratio)
            if spacing_ratio is not None and self._zoom_start_ratio is not None
            else None,
        )

    def _zoom_command_from_motion(self, spacing_ratio: float | None) -> str:
        """Emit only the direction fixed at lock start, and only on movement."""
        if (
            spacing_ratio is None
            or self._zoom_start_ratio is None
            or self._zoom_session_direction is None
        ):
            return "IDLE"

        if self._filtered_zoom_ratio is None:
            self._filtered_zoom_ratio = spacing_ratio
        else:
            self._filtered_zoom_ratio += self.zoom_filter_alpha * (
                spacing_ratio - self._filtered_zoom_ratio
            )

        delta = self._filtered_zoom_ratio - self._zoom_start_ratio
        moved = abs(delta) >= self.zoom_motion_ratio
        if not moved:
            return "IDLE"

        # Advance the reference after a meaningful movement. A stationary hand
        # then becomes IDLE, while a reversal never turns into the opposite zoom.
        self._zoom_start_ratio = self._filtered_zoom_ratio
        if self._zoom_session_direction == "ZOOM_IN" and delta > 0.0:
            return "ZOOM_IN"
        if self._zoom_session_direction == "ZOOM_OUT" and delta < 0.0:
            return "ZOOM_OUT"
        return "IDLE"
