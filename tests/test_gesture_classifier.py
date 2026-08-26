import unittest
import math

from containers.pattern_command.gesture_classifier import GestureClassifier
from test_index_finger import Point


def landmarks(index_open=True, middle_open=False, thumb_active=False, thumb_offset=1.0):
    points = [Point() for _ in range(21)]
    points[17] = Point(-2.0, 0.0)
    points[4] = Point(thumb_offset, 3.0) if thumb_active else Point(0.2, 0.2)
    if index_open:
        points[5] = Point(0.0, 0.0)
        points[6] = Point(0.0, 1.0)
        points[7] = Point(0.0, 2.0)
        points[8] = Point(0.0, 3.0)
    if middle_open:
        points[9] = Point(1.0, 0.0)
        points[10] = Point(1.0, 1.0)
        points[11] = Point(1.0, 2.0)
        points[12] = Point(1.0, 3.0)
    return points


class GestureClassifierTests(unittest.TestCase):
    def test_index_only_is_draw(self):
        classifier = GestureClassifier()
        states = [classifier.update(landmarks()) for _ in range(5)]
        self.assertEqual(states[-1].command, "DRAW")

    def test_two_stable_fingers_are_erase(self):
        classifier = GestureClassifier()
        states = [classifier.update(landmarks(middle_open=True)) for _ in range(9)]
        self.assertEqual(states[-1].command, "ERASE")

    def test_closed_start_locks_zoom_in(self):
        classifier = GestureClassifier()
        for _ in range(7):
            classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        state = classifier.update(landmarks(thumb_active=True, thumb_offset=1.0))
        self.assertEqual(state.command, "ZOOM_IN")
        self.assertEqual(state.mode, "ZOOM")

    def test_open_start_locks_zoom_out(self):
        classifier = GestureClassifier()
        for _ in range(7):
            classifier.update(landmarks(thumb_active=True, thumb_offset=2.2))
        state = classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        self.assertEqual(state.command, "ZOOM_OUT")
        self.assertEqual(state.mode, "ZOOM")

    def test_midpoint_jitter_does_not_create_a_zoom_command(self):
        classifier = GestureClassifier()
        for _ in range(7):
            classifier.update(landmarks(thumb_active=True, thumb_offset=2.2))
        states = [classifier.update(landmarks(thumb_active=True, thumb_offset=offset)) for offset in (2.22, 2.18, 2.21, 2.19)]
        self.assertTrue(all(state.command == "IDLE" for state in states))

    def test_repeating_zoom_in_suppresses_the_opposite_transition(self):
        classifier = GestureClassifier()
        for _ in range(7):
            classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        state = classifier.update(landmarks(thumb_active=True, thumb_offset=1.0))
        self.assertEqual(state.command, "ZOOM_IN")
        states = [classifier.update(landmarks(thumb_active=True, thumb_offset=2.2)) for _ in range(3)]
        self.assertIn("ZOOM_IN", [state.command for state in states])
        for _ in range(3):
            state = classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        self.assertEqual(state.command, "IDLE")
        states = [classifier.update(landmarks(thumb_active=True, thumb_offset=2.2)) for _ in range(3)]
        self.assertIn("ZOOM_IN", [state.command for state in states])

    def test_locked_draw_does_not_switch_to_zoom_when_thumb_appears(self):
        classifier = GestureClassifier()
        for _ in range(5):
            classifier.update(landmarks())
        state = classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        self.assertEqual(state.command, "DRAW")
        self.assertEqual(state.mode, "DRAW")

    def test_zoom_mode_does_not_fall_back_to_draw_when_thumb_tracking_drops(self):
        classifier = GestureClassifier()
        for _ in range(8):
            classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        state = classifier.update(landmarks(thumb_active=False))
        self.assertEqual(state.command, "ZOOM_IN")

    def test_index_closure_releases_a_locked_draw_mode(self):
        classifier = GestureClassifier()
        for _ in range(5):
            state = classifier.update(landmarks())
        self.assertEqual(state.mode, "DRAW")
        for _ in range(3):
            state = classifier.update(landmarks(index_open=False))
        self.assertEqual(state.mode, "IDLE")
        self.assertEqual(state.command, "IDLE")

    def test_slight_index_bend_releases_after_exactly_three_frames(self):
        classifier = GestureClassifier(release_pip_angle_deg=145.0)
        for _ in range(5):
            state = classifier.update(landmarks())
        self.assertEqual(state.mode, "DRAW")

        bent = landmarks()
        angle = math.radians(135.0)
        bent[7] = Point(math.sin(angle), 1.0 - math.cos(angle))
        bent[8] = Point(2.0 * math.sin(angle), 1.0 - 2.0 * math.cos(angle))
        for _ in range(2):
            state = classifier.update(bent)
            self.assertEqual(state.mode, "DRAW")
        state = classifier.update(bent)
        self.assertEqual(state.mode, "IDLE")

    def test_zoom_filter_ignores_small_single_frame_spacing_jitter(self):
        classifier = GestureClassifier(zoom_motion_ratio=0.035, zoom_filter_alpha=0.55)
        for _ in range(7):
            classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
        state = classifier.update(landmarks(thumb_active=True, thumb_offset=0.54))
        self.assertEqual(state.command, "IDLE")

    def test_zoom_confirmation_overlaps_finger_stabilization(self):
        classifier = GestureClassifier()
        states = [
            classifier.update(landmarks(thumb_active=True, thumb_offset=0.5))
            for _ in range(5)
        ]
        self.assertEqual(states[-1].mode, "ZOOM")
        self.assertEqual(states[-1].zoom_session_direction, "ZOOM_IN")


if __name__ == "__main__":
    unittest.main()
