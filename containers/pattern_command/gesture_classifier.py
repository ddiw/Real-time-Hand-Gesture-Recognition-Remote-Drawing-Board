"""Compatibility import; implementation lives in containers/3-pattern-command."""
from __future__ import annotations
import sys
from pathlib import Path
_DIR = Path(__file__).resolve().parents[1] / "3-pattern-command"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
from gesture_classifier import *  # noqa: F401,F403
