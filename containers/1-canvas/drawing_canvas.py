from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
PATTERN_COMMAND_DIR = _HERE.parent / "3-pattern-command"
if not PATTERN_COMMAND_DIR.exists():
    PATTERN_COMMAND_DIR = _HERE / "3-pattern-command"
if str(PATTERN_COMMAND_DIR) not in sys.path:
    sys.path.insert(0, str(PATTERN_COMMAND_DIR))
from gesture_classifier import GestureClassifier


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
WINDOW_NAME = "MediaPipe index finger test"
CANVAS_WINDOW_NAME = "Gesture drawing canvas"
CAPTURE_BUTTON = (20, 180, 220, 235)


class PerformanceMonitor:
    """Rolling end-to-end FPS and MediaPipe latency measurements."""

    def __init__(self, window_size: int = 120) -> None:
        self.frame_times: deque[float] = deque(maxlen=window_size)
        self.inference_ms: deque[float] = deque(maxlen=window_size)
        self.total_started_at = time.perf_counter()
        self.total_frames = 0

    def record(self, inference_ms: float) -> None:
        self.frame_times.append(time.perf_counter())
        self.inference_ms.append(inference_ms)
        self.total_frames += 1

    @property
    def fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        return (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

    @property
    def average_inference_ms(self) -> float:
        return float(np.mean(self.inference_ms)) if self.inference_ms else 0.0

    @property
    def p95_inference_ms(self) -> float:
        return float(np.percentile(self.inference_ms, 95)) if self.inference_ms else 0.0

    def draw(self, frame) -> None:
        message = (
            f"FPS: {self.fps:.1f} | inference: {self.average_inference_ms:.1f} ms "
            f"(p95 {self.p95_inference_ms:.1f})"
        )
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        x = max(10, frame.shape[1] - text_size[0] - 15)
        cv2.rectangle(frame, (x - 5, 8), (frame.shape[1] - 5, 38), (20, 20, 20), -1)
        cv2.putText(
            frame, message, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (80, 240, 240), 1, cv2.LINE_AA,
        )

    def summary(self) -> str:
        elapsed = max(time.perf_counter() - self.total_started_at, 1e-9)
        return (
            f"Performance: {self.total_frames / elapsed:.1f} average FPS, "
            f"MediaPipe {self.average_inference_ms:.1f} ms average / "
            f"{self.p95_inference_ms:.1f} ms p95 (recent {len(self.inference_ms)} frames)"
        )


class DrawingCanvas:
    """Persistent drawing surface rendered with a center-based zoom."""

    def __init__(
        self,
        width: int,
        height: int,
        zoom_step: float = 1.05,
        min_zoom: float = 1.0,
        max_zoom: float = 4.0,
        pen_thickness: int = 4,
        eraser_radius: int = 24,
        min_draw_distance: float = 2.0,
    ) -> None:
        self.width = width
        self.height = height
        self.image = np.full((height, width, 3), 255, dtype=np.uint8)
        self.zoom = 1.0
        self.zoom_step = zoom_step
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.pen_thickness = pen_thickness
        self.eraser_radius = eraser_radius
        self.min_draw_distance = min_draw_distance
        self._previous_draw_point: tuple[int, int] | None = None
        self._cursor_command = "IDLE"
        self._cursor_point: tuple[int, int] | None = None
        self._cursor_direction = (0.0, -1.0)

    def screen_to_canvas(self, point: tuple[int, int]) -> tuple[int, int]:
        center_x = (self.width - 1) / 2.0
        center_y = (self.height - 1) / 2.0
        return (
            round((point[0] - center_x) / self.zoom + center_x),
            round((point[1] - center_y) / self.zoom + center_y),
        )

    def apply(
        self,
        command: str,
        screen_point: tuple[int, int] | None,
        direction: tuple[float, float] | None = None,
    ) -> None:
        self._cursor_command = command if command in ("DRAW", "ERASE") else "IDLE"
        self._cursor_point = screen_point
        if direction is not None and math.hypot(*direction) > 0.0:
            length = math.hypot(*direction)
            self._cursor_direction = (direction[0] / length, direction[1] / length)
        if command == "ZOOM_IN":
            self.zoom = min(self.max_zoom, self.zoom * self.zoom_step)
            self.end_stroke()
            return
        if command == "ZOOM_OUT":
            self.zoom = max(self.min_zoom, self.zoom / self.zoom_step)
            self.end_stroke()
            return
        if command not in ("DRAW", "ERASE") or screen_point is None:
            self.end_stroke()
            return

        point = self.screen_to_canvas(screen_point)
        if not (0 <= point[0] < self.width and 0 <= point[1] < self.height):
            self.end_stroke()
            return

        if command == "ERASE":
            radius = max(1, round(self.eraser_radius / self.zoom))
            cv2.circle(self.image, point, radius, (255, 255, 255), -1, cv2.LINE_AA)
            self.end_stroke()
            return

        thickness = max(1, round(self.pen_thickness / self.zoom))
        if self._previous_draw_point is None:
            cv2.circle(self.image, point, max(1, thickness // 2), (20, 20, 20), -1, cv2.LINE_AA)
        else:
            dx = point[0] - self._previous_draw_point[0]
            dy = point[1] - self._previous_draw_point[1]
            if dx * dx + dy * dy >= (self.min_draw_distance / self.zoom) ** 2:
                cv2.line(self.image, self._previous_draw_point, point, (20, 20, 20), thickness, cv2.LINE_AA)
        self._previous_draw_point = point

    def end_stroke(self) -> None:
        self._previous_draw_point = None

    def hide_cursor(self) -> None:
        self._cursor_command = "IDLE"
        self._cursor_point = None
        self.end_stroke()

    def render(self):
        center = ((self.width - 1) / 2.0, (self.height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, 0.0, self.zoom)
        rendered = cv2.warpAffine(
            self.image,
            matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        cv2.putText(
            rendered, f"Zoom: {self.zoom:.2f}x", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA,
        )
        self._draw_cursor(rendered)
        return rendered

    def _draw_cursor(self, rendered) -> None:
        if self._cursor_point is None:
            return
        point = self._cursor_point
        if self._cursor_command == "ERASE":
            overlay = rendered.copy()
            cv2.circle(overlay, point, self.eraser_radius, (90, 170, 255), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.32, rendered, 0.68, 0.0, rendered)
            cv2.circle(rendered, point, self.eraser_radius, (40, 110, 220), 2, cv2.LINE_AA)
        elif self._cursor_command == "DRAW":
            dx, dy = self._cursor_direction
            perpendicular = (-dy, dx)
            body_end = (round(point[0] - dx * 34), round(point[1] - dy * 34))
            tail_left = (
                round(body_end[0] + perpendicular[0] * 6),
                round(body_end[1] + perpendicular[1] * 6),
            )
            tail_right = (
                round(body_end[0] - perpendicular[0] * 6),
                round(body_end[1] - perpendicular[1] * 6),
            )
            tip_left = (
                round(point[0] - dx * 8 + perpendicular[0] * 6),
                round(point[1] - dy * 8 + perpendicular[1] * 6),
            )
            tip_right = (
                round(point[0] - dx * 8 - perpendicular[0] * 6),
                round(point[1] - dy * 8 - perpendicular[1] * 6),
            )
            pen_shape = np.array(
                [tail_left, tip_left, point, tip_right, tail_right], dtype=np.int32,
            )
            cv2.fillConvexPoly(rendered, pen_shape, (40, 190, 245), cv2.LINE_AA)
            cv2.polylines(rendered, [pen_shape], True, (35, 55, 70), 2, cv2.LINE_AA)
            cv2.circle(rendered, point, 2, (20, 20, 20), -1, cv2.LINE_AA)


class CaptureRequest:
    requested = False

    def on_mouse(self, event, x, y, _flags, _parameter) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        left, top, right, bottom = CAPTURE_BUTTON
        if left <= x <= right and top <= y <= bottom:
            self.requested = True


def drawing_area_for_frame(
    frame_width: int,
    frame_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    """Largest centered camera rectangle with the target canvas aspect ratio."""
    target_aspect = canvas_width / canvas_height
    frame_aspect = frame_width / frame_height
    if frame_aspect > target_aspect:
        area_height = frame_height
        area_width = round(area_height * target_aspect)
    else:
        area_width = frame_width
        area_height = round(area_width / target_aspect)
    left = (frame_width - area_width) // 2
    top = (frame_height - area_height) // 2
    return left, top, left + area_width, top + area_height


def camera_to_canvas_point(
    point: tuple[int, int],
    drawing_area: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int] | None:
    left, top, right, bottom = drawing_area
    if not (left <= point[0] < right and top <= point[1] < bottom):
        return None
    x = round((point[0] - left) * (canvas_width - 1) / max(1, right - left - 1))
    y = round((point[1] - top) * (canvas_height - 1) / max(1, bottom - top - 1))
    return x, y


def draw_drawing_area(frame, area: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = area
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    overlay[top:bottom, left:right] = frame[top:bottom, left:right]
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, frame)
    cv2.rectangle(frame, (left, top), (right - 1, bottom - 1), (60, 220, 255), 2)
    cv2.putText(
        frame, "PHONE DRAWING AREA", (left + 8, min(bottom - 10, top + 25)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 255), 2, cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe index-finger rule test")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("containers/2-vision-analysis/models/hand_landmarker.task"),
    )
    parser.add_argument("--pip-angle-open", type=float, default=120.0)
    parser.add_argument(
        "--release-pip-angle", type=float, default=145.0,
        help="PIP angle below this value counts toward the 3-frame mode release",
    )
    parser.add_argument(
        "--zoom-motion", type=float, default=0.05,
        help="Normalized spacing movement required for one zoom event",
    )
    parser.add_argument(
        "--zoom-filter-alpha", type=float, default=0.75,
        help="Zoom spacing smoothing; lower is steadier, higher is more responsive",
    )
    parser.add_argument("--output", type=Path, default=Path("captures"))
    parser.add_argument("--canvas-width", type=int, default=360)
    parser.add_argument("--canvas-height", type=int, default=640)
    return parser.parse_args()


def draw_landmarks(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(item.x * width), int(item.y * height)) for item in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (80, 210, 80), 2)
    for index, point in enumerate(points):
        color = (0, 200, 255) if index in (0, 5, 6, 7, 8, 9, 17) else (255, 180, 30)
        cv2.circle(frame, point, 4, color, -1)


def draw_capture_button(frame) -> None:
    left, top, right, bottom = CAPTURE_BUTTON
    cv2.rectangle(frame, (left, top), (right, bottom), (30, 170, 240), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)
    cv2.putText(
        frame, "CAPTURE", (left + 42, top + 36),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA,
    )


def save_capture(output_dir: Path, frame, landmarks, world_landmarks, state) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = output_dir / f"capture_{capture_id}.jpg"
    data_path = output_dir / f"capture_{capture_id}.json"
    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"Could not save capture image: {image_path}")

    payload = {
        "captured_at": datetime.now().astimezone().isoformat(),
        "classification": asdict(state) if state is not None else None,
        "landmarks": [
            {"index": index, "x": point.x, "y": point.y, "z": point.z}
            for index, point in enumerate(landmarks or [])
        ],
        "world_landmarks": [
            {"index": index, "x": point.x, "y": point.y, "z": point.z}
            for index, point in enumerate(world_landmarks or [])
        ],
    }
    data_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return image_path, data_path


def main() -> None:
    import mediapipe as mp

    args = parse_args()
    if args.canvas_width <= 0 or args.canvas_height <= 0:
        raise ValueError("Canvas width and height must be positive")
    if not args.model.exists():
        raise FileNotFoundError(
            f"MediaPipe model not found: {args.model}\n"
            "Run download_model.ps1 first."
        )

    classifier = GestureClassifier(
        open_pip_angle_deg=args.pip_angle_open,
        release_pip_angle_deg=args.release_pip_angle,
        zoom_motion_ratio=args.zoom_motion,
        zoom_filter_alpha=args.zoom_filter_alpha,
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    print(
        "Camera: "
        f"{int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"reported {camera.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )

    started_at = time.monotonic()
    previous_timestamp_ms = -1
    capture_request = CaptureRequest()
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, capture_request.on_mouse)
    capture_notice = ""
    capture_notice_until = 0.0
    canvas = None
    performance = PerformanceMonitor()
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera frame could not be read")

                frame = cv2.flip(frame, 1)
                drawing_area = drawing_area_for_frame(
                    frame.shape[1], frame.shape[0], args.canvas_width, args.canvas_height,
                )
                if canvas is None:
                    canvas = DrawingCanvas(args.canvas_width, args.canvas_height)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                elapsed_timestamp_ms = int((time.monotonic() - started_at) * 1000)
                # MediaPipe VIDEO mode rejects duplicated timestamps. A fast webcam
                # can produce multiple frames within one millisecond.
                timestamp_ms = max(previous_timestamp_ms + 1, elapsed_timestamp_ms)
                previous_timestamp_ms = timestamp_ms
                inference_started_at = time.perf_counter()
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
                performance.record(inference_ms)

                current_landmarks = None
                current_world_landmarks = None
                current_state = None
                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    world_landmarks = result.hand_world_landmarks[0]
                    state = classifier.update(world_landmarks)
                    current_landmarks = landmarks
                    current_world_landmarks = world_landmarks
                    current_state = state
                    draw_landmarks(frame, landmarks)
                    camera_index_tip = (
                        int(landmarks[8].x * frame.shape[1]),
                        int(landmarks[8].y * frame.shape[0]),
                    )
                    camera_index_pip = (
                        int(landmarks[6].x * frame.shape[1]),
                        int(landmarks[6].y * frame.shape[0]),
                    )
                    index_tip = camera_to_canvas_point(
                        camera_index_tip,
                        drawing_area,
                        args.canvas_width,
                        args.canvas_height,
                    )
                    index_pip = camera_to_canvas_point(
                        camera_index_pip,
                        drawing_area,
                        args.canvas_width,
                        args.canvas_height,
                    )
                    direction = (
                        (index_tip[0] - index_pip[0], index_tip[1] - index_pip[1])
                        if index_tip is not None and index_pip is not None
                        else None
                    )
                    canvas.apply(
                        state.command,
                        index_tip,
                        direction,
                    )
                    lines = (
                        f"Command: {state.command}",
                        f"Locked mode: {state.mode} (release {state.release_frames}/3)",
                        f"Index / middle / ring / pinky: {state.index.stable_label} / {state.middle.stable_label} / {state.ring.stable_label} / {state.pinky.stable_label}",
                        f"PIP angle (index / middle): {state.index.pip_angle_deg:.1f} / {state.middle.pip_angle_deg:.1f} deg",
                        f"thumb active / zoom mode: {state.thumb_active} / {state.zoom_mode_active}",
                        f"thumb-index spacing / delta: {state.thumb_index_spacing_ratio or 0.0:.2f} / {state.zoom_delta or 0.0:+.2f}",
                        f"zoom session direction: {state.zoom_session_direction or '-'}",
                    )
                    color = (40, 220, 40) if state.command != "IDLE" else (40, 80, 255)
                else:
                    classifier.reset()
                    canvas.hide_cursor()
                    lines = ("Hand not detected",)
                    color = (40, 80, 255)

                for row, message in enumerate(lines):
                    cv2.putText(
                        frame, message, (20, 35 + row * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
                    )
                draw_drawing_area(frame, drawing_area)
                draw_capture_button(frame)
                cv2.putText(
                    frame, "Click CAPTURE or press SPACE | Q / ESC: quit", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA,
                )
                if capture_notice and time.monotonic() < capture_notice_until:
                    cv2.putText(
                        frame, capture_notice, (20, 270), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (40, 240, 240), 2, cv2.LINE_AA,
                    )
                performance.draw(frame)
                cv2.imshow(WINDOW_NAME, frame)
                cv2.imshow(CANVAS_WINDOW_NAME, canvas.render())
                key = cv2.waitKey(1) & 0xFF
                if capture_request.requested or key == ord(" "):
                    capture_request.requested = False
                    image_path, _ = save_capture(
                        args.output, frame, current_landmarks, current_world_landmarks, current_state,
                    )
                    capture_notice = f"Saved: {image_path.name}"
                    capture_notice_until = time.monotonic() + 2.0
                    print(f"Saved capture: {image_path.resolve()}")
                if key in (ord("q"), 27):
                    break
    finally:
        print(performance.summary())
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
