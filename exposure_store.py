"""Carry the exposure setting between sessions via a small JSON file.

Why a file: uEye (IDS) cameras reset all settings to driver defaults on
is_InitCamera — the default exposure is ~1/framerate (e.g. ~66 ms), NOT what
the previous program set. So "read the sensor's current exposure" cannot carry
a value from the visualizer into oct_scan across process restarts. Instead the
visualizer records its exposure here every time it changes, and oct_scan reads
it back when --exposure is not given.

The stored value is the non-binned-equivalent exposure in microseconds —
directly usable for full-resolution captures.
"""

import json
import time
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "last_exposure.json"


def save_exposure(exposure_us: float):
    """Record the current session exposure (best-effort; never raises)."""
    try:
        _PATH.write_text(json.dumps(
            {"exposure_us": float(exposure_us), "saved_at": time.time()}))
    except OSError:
        pass


def load_exposure():
    """Return (exposure_us, age_description) from the last session, or None."""
    try:
        data = json.loads(_PATH.read_text())
        us = float(data["exposure_us"])
    except (OSError, ValueError, KeyError):
        return None
    age_s = time.time() - data.get("saved_at", 0)
    if age_s < 3600:
        age = f"{age_s / 60:.0f} min ago"
    elif age_s < 86400:
        age = f"{age_s / 3600:.1f} h ago"
    else:
        age = f"{age_s / 86400:.0f} days ago"
    return us, age
