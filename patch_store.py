"""Carry the visualizer's selected patch between sessions via a JSON file.

The visualizer records the two-click patch here (in FULL-RESOLUTION sensor
coordinates — preview coordinates are scaled up by the preview binning), and
oct_crop_scan.py uses it as the default camera ROI, so the region you picked
on screen is exactly the region the cropped scan reads out.
"""

import json
import time
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "last_patch.json"


def save_patch(x, y, w, h, binning=1):
    """Record a patch given in preview coords at `binning` (best-effort)."""
    try:
        _PATH.write_text(json.dumps({
            "x": int(x * binning), "y": int(y * binning),
            "w": int(w * binning), "h": int(h * binning),
            "saved_at": time.time()}))
    except OSError:
        pass


def load_patch():
    """Return ((x, y, w, h) full-res sensor coords, age string) or None."""
    try:
        d = json.loads(_PATH.read_text())
        rect = (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))
    except (OSError, ValueError, KeyError):
        return None
    age_s = time.time() - d.get("saved_at", 0)
    if age_s < 3600:
        age = f"{age_s / 60:.0f} min ago"
    elif age_s < 86400:
        age = f"{age_s / 3600:.1f} h ago"
    else:
        age = f"{age_s / 86400:.0f} days ago"
    return rect, age
