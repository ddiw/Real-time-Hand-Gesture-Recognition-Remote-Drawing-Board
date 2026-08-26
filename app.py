"""Backward-compatible local entry point for the container-owned canvas app."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent / "containers/1-canvas/drawing_canvas.py"
_SPEC = importlib.util.spec_from_file_location("drawing_canvas_local", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load canvas application: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DrawingCanvas = _MODULE.DrawingCanvas
PerformanceMonitor = _MODULE.PerformanceMonitor
camera_to_canvas_point = _MODULE.camera_to_canvas_point
drawing_area_for_frame = _MODULE.drawing_area_for_frame
draw_drawing_area = _MODULE.draw_drawing_area
main = _MODULE.main

if __name__ == "__main__":
    main()
