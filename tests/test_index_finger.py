import unittest
from dataclasses import dataclass

from containers.pattern_command.index_finger import IndexFingerClassifier, joint_angle


@dataclass
class Point:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def open_index_landmarks():
    points = [Point() for _ in range(21)]
    points[0] = Point(0.0, 0.0)
    points[9] = Point(0.0, 1.0)
    points[5] = Point(0.0, 1.0)
    points[6] = Point(0.0, 2.0)
    points[7] = Point(0.0, 3.0)
    points[8] = Point(0.0, 4.0)
    return points


def closed_index_landmarks():
    points = open_index_landmarks()
    points[7] = Point(1.0, 2.0)
    points[8] = Point(1.0, 1.0)
    return points


class IndexFingerClassifierTests(unittest.TestCase):
    def test_joint_angle_for_straight_finger_is_180_degrees(self):
        points = open_index_landmarks()
        self.assertAlmostEqual(joint_angle(points[5], points[6], points[7]), 180.0)

    def test_open_state_requires_stable_frame_history(self):
        classifier = IndexFingerClassifier()
        states = [classifier.update(open_index_landmarks()) for _ in range(5)]
        self.assertEqual(states[0].raw_label, "OPEN")
        self.assertEqual(states[0].stable_label, "CLOSED")
        self.assertEqual(states[-1].stable_label, "OPEN")

    def test_bent_finger_is_closed(self):
        classifier = IndexFingerClassifier()
        states = [classifier.update(closed_index_landmarks()) for _ in range(5)]
        self.assertLess(states[-1].pip_angle_deg, 150.0)
        self.assertEqual(states[-1].stable_label, "CLOSED")

    def test_pip_angle_alone_is_the_open_rule(self):
        classifier = IndexFingerClassifier(open_pip_angle_deg=120.0)
        points = open_index_landmarks()
        points[7] = Point(1.0, 3.7)
        states = [classifier.update(points) for _ in range(5)]
        self.assertGreaterEqual(states[-1].pip_angle_deg, 120.0)
        self.assertEqual(states[-1].stable_label, "OPEN")


if __name__ == "__main__":
    unittest.main()
