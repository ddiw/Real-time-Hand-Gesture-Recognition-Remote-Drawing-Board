import unittest

from containers.pattern_command.gesture_classifier import GestureClassifier
from test_index_finger import Point


FINGER_COLUMNS = ((5, 0.0), (9, 1.0), (13, 2.0), (17, -2.0))


def landmarks(count=1, order=None):
    """World landmarks with `count` fingers extended, index -> pinky.

    `order` overrides which fingers open, so out-of-order postures such as a
    lone pinky can be exercised.
    """
    points = [Point() for _ in range(21)]
    points[17] = Point(-2.0, 0.0)
    points[4] = Point(0.2, 0.2)
    open_fingers = order if order is not None else range(count)
    for finger in open_fingers:
        base, x = FINGER_COLUMNS[finger]
        for offset, y in enumerate((0.0, 1.0, 2.0, 3.0)):
            points[base + offset] = Point(x, y)
    return points


def screen(x=0.5):
    """Screen landmarks with a 0.2-wide palm centred on `x` (ratio == 5x)."""
    points = [Point(x, 0.5) for _ in range(21)]
    points[0] = Point(x, 0.6)
    points[5] = Point(x + 0.1, 0.5)
    points[17] = Point(x - 0.1, 0.5)
    return points


def zoom_classifier():
    return GestureClassifier(zoom_deadzone_ratio=0.15, zoom_filter_alpha=1.0)


def settle(classifier, count, x=0.5, frames=5):
    state = None
    for _ in range(frames):
        state = classifier.update(landmarks(count), screen(x))
    return state


class FingerCountCommandTests(unittest.TestCase):
    def test_one_finger_is_draw(self):
        self.assertEqual(settle(GestureClassifier(), 1).command, "DRAW")

    def test_two_fingers_are_erase(self):
        self.assertEqual(settle(GestureClassifier(), 2).command, "ERASE")

    def test_four_fingers_have_no_command(self):
        self.assertEqual(settle(GestureClassifier(), 4).command, "IDLE")

    def test_out_of_order_posture_is_ignored(self):
        classifier = GestureClassifier()
        state = None
        for _ in range(5):
            state = classifier.update(landmarks(order=(3,)), screen())
        self.assertEqual(state.finger_count, 1)
        self.assertEqual(state.command, "IDLE")

    def test_posture_change_switches_command_without_a_fist(self):
        classifier = GestureClassifier()
        self.assertEqual(settle(classifier, 1).command, "DRAW")
        self.assertEqual(settle(classifier, 2).command, "ERASE")
        self.assertEqual(settle(classifier, 1).command, "DRAW")


class FistClearTests(unittest.TestCase):
    def test_fist_clears_once_and_then_idles(self):
        classifier = GestureClassifier()
        settle(classifier, 1)
        first = classifier.update(landmarks(0), screen())
        for _ in range(4):
            first = classifier.update(landmarks(0), screen())
            if first.command == "CLEAR":
                break
        self.assertEqual(first.command, "CLEAR")
        held = [classifier.update(landmarks(0), screen()).command for _ in range(4)]
        self.assertTrue(all(command == "IDLE" for command in held))

    def test_a_new_hand_does_not_clear_while_the_stabilizer_warms_up(self):
        classifier = GestureClassifier()
        commands = [classifier.update(landmarks(1), screen()).command for _ in range(5)]
        self.assertNotIn("CLEAR", commands)

    def test_reopening_the_hand_arms_the_next_clear(self):
        classifier = GestureClassifier()
        settle(classifier, 1)
        settle(classifier, 0)
        settle(classifier, 1)
        second = [classifier.update(landmarks(0), screen()).command for _ in range(5)]
        self.assertEqual(second.count("CLEAR"), 1)


class ThreeFingerZoomTests(unittest.TestCase):
    def test_holding_left_of_neutral_keeps_zooming_in(self):
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        held = [classifier.update(landmarks(3), screen(0.44)).command for _ in range(5)]
        self.assertTrue(all(command == "ZOOM_IN" for command in held))

    def test_holding_right_of_neutral_keeps_zooming_out(self):
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        held = [classifier.update(landmarks(3), screen(0.56)).command for _ in range(5)]
        self.assertTrue(all(command == "ZOOM_OUT" for command in held))

    def test_returning_to_neutral_stops_without_zooming_out(self):
        """The ratchet bug: coming back to repeat a zoom-in must not zoom out."""
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        self.assertEqual(classifier.update(landmarks(3), screen(0.44)).command, "ZOOM_IN")
        back = [
            classifier.update(landmarks(3), screen(x)).command
            for x in (0.47, 0.49, 0.50)
        ]
        self.assertTrue(all(command == "IDLE" for command in back))
        self.assertEqual(classifier.update(landmarks(3), screen(0.44)).command, "ZOOM_IN")

    def test_crossing_to_the_far_side_reverses_direction(self):
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        self.assertEqual(classifier.update(landmarks(3), screen(0.44)).command, "ZOOM_IN")
        self.assertEqual(classifier.update(landmarks(3), screen(0.56)).command, "ZOOM_OUT")

    def test_small_jitter_does_not_zoom(self):
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        jitter = [
            classifier.update(landmarks(3), screen(x)).command
            for x in (0.505, 0.497, 0.503, 0.499)
        ]
        self.assertTrue(all(command == "IDLE" for command in jitter))

    def test_dropping_the_posture_recaptures_neutral(self):
        classifier = zoom_classifier()
        settle(classifier, 3, x=0.50)
        self.assertEqual(classifier.update(landmarks(3), screen(0.44)).command, "ZOOM_IN")
        settle(classifier, 1, x=0.44)
        settle(classifier, 3, x=0.44)
        # Neutral is now the new resting spot, so holding still does nothing.
        held = [classifier.update(landmarks(3), screen(0.44)).command for _ in range(5)]
        self.assertTrue(all(command == "IDLE" for command in held))

    def test_zoom_needs_screen_landmarks(self):
        classifier = zoom_classifier()
        state = None
        for _ in range(6):
            state = classifier.update(landmarks(3))
        self.assertFalse(state.zoom_active)
        self.assertEqual(state.command, "IDLE")

    def test_offset_is_measured_in_palm_widths(self):
        """A hand twice as close needs twice the pixel offset to leave neutral."""
        far = zoom_classifier()
        settle(far, 3, x=0.50)
        self.assertEqual(far.update(landmarks(3), screen(0.46)).command, "ZOOM_IN")

        near = zoom_classifier()
        wide = [Point(0.5, 0.5) for _ in range(21)]
        wide[0], wide[5], wide[17] = Point(0.5, 0.7), Point(0.7, 0.5), Point(0.3, 0.5)
        for _ in range(5):
            near.update(landmarks(3), wide)
        moved = [Point(p.x - 0.04, p.y) for p in wide]
        self.assertEqual(near.update(landmarks(3), moved).command, "IDLE")


if __name__ == "__main__":
    unittest.main()
