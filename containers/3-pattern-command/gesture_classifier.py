from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Sequence

from index_finger import IndexFingerClassifier, IndexFingerState, Landmark, distance_3d


@dataclass(frozen=True)
class GestureState:
    command: str
    mode: str
    finger_count: int
    index: IndexFingerState
    middle: IndexFingerState
    ring: IndexFingerState
    pinky: IndexFingerState
    thumb_active: bool
    zoom_active: bool
    horizontal_ratio: float | None
    horizontal_delta: float | None
    zoom_direction: str | None


class GestureClassifier:
    """Container C rule engine driven by the extended finger count.

    Postures are counted in the natural order index -> middle -> ring -> pinky,
    so only a prefix of that order produces a command. The thumb is never part
    of the count; it only reports as a diagnostic value.

    - 0 fingers (fist): ``CLEAR`` once per fist, then IDLE until the hand opens
    - 1 finger: ``DRAW``
    - 2 fingers: ``ERASE``
    - 3 fingers: zoom, controlled by where the hand sits relative to the
      neutral point captured when the posture began
    - anything else: ``IDLE``

    There is no mode lock. A posture change takes effect on the frame the new
    posture stabilizes, and the fist is the only way to reset the canvas.
    """

    def __init__(
        self,
        open_pip_angle_deg: float = 120.0,
        thumb_active_ratio: float = 0.65,
        zoom_deadzone_ratio: float = 0.15,
        zoom_filter_alpha: float = 0.6,
        warmup_frames: int = 13,
        clear_hold_frames: int = 13,
        finger_window: int = 5,
        finger_open_votes: int = 4,
        **_legacy: object,
    ) -> None:
        if not 0.0 < zoom_filter_alpha <= 1.0:
            raise ValueError("zoom_filter_alpha must be within (0, 1]")
        if zoom_deadzone_ratio <= 0.0:
            raise ValueError("zoom_deadzone_ratio must be positive")
        finger = dict(
            open_pip_angle_deg=open_pip_angle_deg,
            window_size=finger_window,
            required_open_votes=finger_open_votes,
        )
        self.index_classifier = IndexFingerClassifier(**finger, finger_indices=(5, 6, 7, 8))
        self.middle_classifier = IndexFingerClassifier(**finger, finger_indices=(9, 10, 11, 12))
        self.ring_classifier = IndexFingerClassifier(**finger, finger_indices=(13, 14, 15, 16))
        self.pinky_classifier = IndexFingerClassifier(**finger, finger_indices=(17, 18, 19, 20))
        self.thumb_active_ratio = thumb_active_ratio
        self.zoom_deadzone_ratio = zoom_deadzone_ratio
        self.zoom_filter_alpha = zoom_filter_alpha
        self.warmup_frames = max(warmup_frames, finger_window)
        self.clear_hold_frames = max(1, clear_hold_frames)
        self._frames = 0
        self._fist_frames = 0
        self._clear_sent = False
        self._zoom_neutral: float | None = None
        self._filtered_ratio: float | None = None

    def reset(self) -> None:
        self.index_classifier.reset()
        self.middle_classifier.reset()
        self.ring_classifier.reset()
        self.pinky_classifier.reset()
        self._frames = 0
        self._fist_frames = 0
        self._clear_sent = False
        self._zoom_neutral = None
        self._filtered_ratio = None

    def update(
        self,
        landmarks: Sequence[Landmark],
        screen_landmarks: Sequence[Landmark] | None = None,
    ) -> GestureState:
        self._frames += 1
        index = self.index_classifier.update(landmarks)
        middle = self.middle_classifier.update(landmarks)
        ring = self.ring_classifier.update(landmarks)
        pinky = self.pinky_classifier.update(landmarks)

        opened = [
            finger.stable_label == "OPEN"
            for finger in (index, middle, ring, pinky)
        ]
        finger_count = sum(opened)
        # Only a prefix of index -> middle -> ring -> pinky counts. A stray
        # pinky is not "one finger"; it is an unrecognized posture.
        counted_in_order = opened == [True] * finger_count + [False] * (4 - finger_count)

        palm_width = distance_3d(landmarks[5], landmarks[17])
        thumb_active = (
            palm_width > 0.0
            and distance_3d(landmarks[4], landmarks[5]) / palm_width >= self.thumb_active_ratio
        )

        horizontal_ratio = self._horizontal_ratio(screen_landmarks)
        zoom_active = counted_in_order and finger_count == 3 and horizontal_ratio is not None

        if finger_count == 0:
            self._fist_frames += 1
        else:
            self._fist_frames = 0
            self._clear_sent = False
        if not zoom_active:
            # Dropping the posture forgets neutral, so the next three-finger
            # gesture re-anchors wherever the hand happens to be.
            self._zoom_neutral = None
            self._filtered_ratio = None

        horizontal_delta = None
        if finger_count == 0:
            # Two separate guards, because they cover different accidents.
            # Warm-up covers a hand that has just appeared: the stabilizer
            # reports CLOSED until its window fills, so an arriving hand reads
            # as a fist. The hold covers a hand already in frame, where a
            # momentary tracking loss mid-gesture used to wipe the canvas.
            warm = self._frames >= self.warmup_frames
            held = self._fist_frames >= self.clear_hold_frames
            mode = "CLEAR" if warm else "IDLE"
            if warm and held and not self._clear_sent:
                command = "CLEAR"
                self._clear_sent = True
            else:
                command = "IDLE"
        elif not counted_in_order:
            mode, command = "IDLE", "IDLE"
        elif finger_count == 1:
            mode, command = "DRAW", "DRAW"
        elif finger_count == 2:
            mode, command = "ERASE", "ERASE"
        elif zoom_active:
            mode = "ZOOM"
            command, horizontal_delta = self._zoom_command(horizontal_ratio)
        else:
            mode, command = "IDLE", "IDLE"

        return GestureState(
            command=command,
            mode=mode,
            finger_count=finger_count,
            index=index,
            middle=middle,
            ring=ring,
            pinky=pinky,
            thumb_active=thumb_active,
            zoom_active=zoom_active,
            horizontal_ratio=horizontal_ratio,
            horizontal_delta=horizontal_delta,
            zoom_direction=command if command in ("ZOOM_IN", "ZOOM_OUT") else None,
        )

    def _horizontal_ratio(self, screen_landmarks: Sequence[Landmark] | None) -> float | None:
        """Palm centre x expressed in palm widths, from screen-space landmarks.

        World landmarks are hand-centred, so they never register the hand
        travelling across the frame. Normalizing by the on-screen palm width
        keeps the value stable as the hand moves toward or away from the camera.
        """
        if screen_landmarks is None or len(screen_landmarks) != 21:
            return None
        wrist, index_mcp, pinky_mcp = screen_landmarks[0], screen_landmarks[5], screen_landmarks[17]
        palm_width = hypot(index_mcp.x - pinky_mcp.x, index_mcp.y - pinky_mcp.y)
        if palm_width <= 0.0:
            return None
        return (wrist.x + index_mcp.x + pinky_mcp.x) / 3.0 / palm_width

    def _zoom_command(self, horizontal_ratio: float | None) -> tuple[str, float | None]:
        """Hold left of neutral to keep zooming in, right of it to zoom out.

        The neutral point is captured once, when the three-finger posture
        begins, and never moves while the posture is held. A ratcheting scheme
        that advanced the reference on every event made the second zoom
        impossible: the hand ran out of frame, and bringing it back to repeat
        the motion read as travel the other way. Here, returning to neutral
        simply stops; only crossing to the far side reverses the direction.
        """
        if horizontal_ratio is None:
            return "IDLE", None

        if self._filtered_ratio is None:
            self._filtered_ratio = horizontal_ratio
        else:
            self._filtered_ratio += self.zoom_filter_alpha * (
                horizontal_ratio - self._filtered_ratio
            )
        if self._zoom_neutral is None:
            self._zoom_neutral = self._filtered_ratio
            return "IDLE", 0.0

        offset = self._filtered_ratio - self._zoom_neutral
        if abs(offset) < self.zoom_deadzone_ratio:
            return "IDLE", offset
        return ("ZOOM_IN" if offset < 0.0 else "ZOOM_OUT"), offset
