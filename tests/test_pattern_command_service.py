import importlib.util
import sys
import unittest
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1] / "containers/3-pattern-command"
sys.path.insert(0, str(SERVICE_DIR))
spec = importlib.util.spec_from_file_location("pattern_command_service", SERVICE_DIR / "app.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def landmarks(index_open=True):
    points = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
    points[17] = {"x": -2.0, "y": 0.0, "z": 0.0}
    points[4] = {"x": 0.2, "y": 0.2, "z": 0.0}
    if index_open:
        for index, y in zip((5, 6, 7, 8), (0.0, 1.0, 2.0, 3.0)):
            points[index] = {"x": 0.0, "y": y, "z": 0.0}
    return points


class PatternCommandServiceTests(unittest.TestCase):
    def test_prd_packet_becomes_draw_command_with_normalized_pointer(self):
        processor = module.CommandProcessor()
        packet = {
            "session_id": "phone-1", "seq": 7, "capture_ts": 1000,
            "processed_ts": 1012, "hand_present": True,
            "landmarks": landmarks(), "world_landmarks": landmarks(),
        }
        states = [processor.process(packet) for _ in range(5)]
        self.assertEqual(states[-1]["command"], "DRAW")
        self.assertEqual(states[-1]["index_tip"], {"x": 0.0, "y": 3.0})
        self.assertEqual(states[-1]["inference_ms"], 12)

    def test_absent_hand_resets_session(self):
        processor = module.CommandProcessor()
        packet = {"session_id": "phone-1", "seq": 1, "hand_present": False}
        state = processor.process(packet)
        self.assertEqual(state["command"], "IDLE")
        self.assertNotIn("index_tip", state)


if __name__ == "__main__":
    unittest.main()
