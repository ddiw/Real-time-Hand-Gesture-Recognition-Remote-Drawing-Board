import unittest

import numpy as np

from app import (
    DrawingCanvas,
    PerformanceMonitor,
    camera_to_canvas_point,
    drawing_area_for_frame,
)


class DrawingCanvasTests(unittest.TestCase):
    def test_draw_connects_index_tip_points(self):
        canvas = DrawingCanvas(100, 100, min_draw_distance=0)
        canvas.apply("DRAW", (20, 20))
        canvas.apply("DRAW", (80, 80))
        self.assertTrue(np.any(canvas.image[45:56, 45:56] < 255))

    def test_erase_clears_the_target_location(self):
        canvas = DrawingCanvas(100, 100)
        canvas.image[:] = 0
        canvas.apply("ERASE", (50, 50))
        self.assertTrue(np.all(canvas.image[50, 50] == 255))

    def test_zoom_changes_only_for_zoom_events(self):
        canvas = DrawingCanvas(100, 100)
        canvas.apply("IDLE", (50, 50))
        self.assertEqual(canvas.zoom, 1.0)
        canvas.apply("ZOOM_IN", (50, 50))
        self.assertAlmostEqual(canvas.zoom, 1.05)
        canvas.apply("IDLE", (50, 50))
        self.assertAlmostEqual(canvas.zoom, 1.05)
        canvas.apply("ZOOM_OUT", (50, 50))
        self.assertAlmostEqual(canvas.zoom, 1.0)

    def test_zoom_out_never_exposes_undrawable_space(self):
        canvas = DrawingCanvas(100, 100)
        for _ in range(20):
            canvas.apply("ZOOM_OUT", (50, 50))
        self.assertEqual(canvas.zoom, 1.0)

    def test_zoom_uses_canvas_center(self):
        canvas = DrawingCanvas(101, 101)
        canvas.zoom = 2.0
        self.assertEqual(canvas.screen_to_canvas((50, 50)), (50, 50))
        self.assertEqual(canvas.screen_to_canvas((70, 50)), (60, 50))

    def test_eraser_cursor_is_visible_but_not_saved_to_canvas(self):
        canvas = DrawingCanvas(100, 100)
        canvas.apply("ERASE", (50, 50))
        stored = canvas.image.copy()
        rendered = canvas.render()
        self.assertFalse(np.array_equal(rendered[50, 50], stored[50, 50]))
        self.assertTrue(np.array_equal(canvas.image, stored))

    def test_pen_cursor_follows_finger_direction(self):
        horizontal = DrawingCanvas(100, 100)
        horizontal.apply("DRAW", (50, 50), (1.0, 0.0))
        vertical = DrawingCanvas(100, 100)
        vertical.apply("DRAW", (50, 50), (0.0, 1.0))
        self.assertFalse(np.array_equal(horizontal.render(), vertical.render()))


class PerformanceMonitorTests(unittest.TestCase):
    def test_summary_reports_recorded_inference(self):
        monitor = PerformanceMonitor()
        monitor.record(10.0)
        monitor.record(20.0)
        self.assertAlmostEqual(monitor.average_inference_ms, 15.0)
        self.assertIn("MediaPipe 15.0 ms", monitor.summary())


class DrawingAreaMappingTests(unittest.TestCase):
    def test_portrait_phone_area_is_centered_in_landscape_camera(self):
        area = drawing_area_for_frame(640, 480, 360, 640)
        self.assertEqual(area, (185, 0, 455, 480))

    def test_phone_area_edges_map_to_full_canvas(self):
        area = (185, 0, 455, 480)
        self.assertEqual(camera_to_canvas_point((185, 0), area, 360, 640), (0, 0))
        self.assertEqual(camera_to_canvas_point((454, 479), area, 360, 640), (359, 639))
        self.assertIsNone(camera_to_canvas_point((184, 200), area, 360, 640))


if __name__ == "__main__":
    unittest.main()
