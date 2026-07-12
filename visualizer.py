#!/usr/bin/env python3
"""
visualizer.py — live camera + stage viewer with a coherence-envelope panel.

Successor to live_view_coherence.py with a redesigned interface: dark theme,
Avenir (or the closest geometric face installed) for all text via Pillow, stat
tiles for the live readouts, and a cleaner plot.

Select a rectangular patch with two clicks; it is shown magnified in the panel
(so fringes and speckle are legible at sensor resolution) and its interference
amplitude (the compute_mean_diff measure: mean |frame - local mean| over the
patch) is plotted against stage position. The local (DC) mean is positional — patches captured
within +/- --window-mm of the current stage position — so irregular stage
motion is handled naturally, and the plot keeps exactly one value per position.
Once the envelope crosses half-max on both sides of the peak, the coherence
length (FWHM after floor subtraction) and the peak location are shown.

Controls:
  click x2   : Select patch (first click anchors a corner, second finalizes)
  wheel      : (over the plot) zoom the position axis around the cursor
  left-drag  : (over the plot) pan the position axis
  f          : Fine scan — once coarse alignment has the peak roughly in view,
               steps +/- 0.5 mm around the current position at 40 um, saves a
               contrast-vs-position plot (PNG + CSV) and moves the stage to the
               coherence peak (needs a patch; f again cancels)
  m          : Move to a typed position — type the mm and press Enter (a
               leading + or - moves relative to the current position; Esc
               cancels). Outside the visualizer, use: python stage.py --to 3.75
  i          : Toggle auto-contrast on the magnified patch inset (percentile
               stretch — makes faint fringes visible, but exaggerates contrast
               and hides how close the patch is to saturation)
  a          : Autoscale the plot (undo zoom/pan)
  c          : Clear the patch selection and the plot
  r          : Reset (clear) the plot data but keep the patch
  +/-        : Increase/decrease exposure
  up / w     : Move stage by +coarse step (hold to sweep)
  down / s   : Move stage by -coarse step (hold to sweep)
  e / d      : Move stage by +/- fine step (coarse / FINE_RATIO)
  [ / ]      : Decrease / increase the coarse step size
  q          : Quit

Usage:
    python visualizer.py
    python visualizer.py --step 0.1 --window-mm 0.2
    python visualizer.py --binning 4 --exposure 10000
    python visualizer.py --fine-range-mm 0.3 --fine-step-mm 0.02
"""

import cv2
import argparse
import numpy as np
from collections import deque
import time

# stage.py is importable without the drivers installed (pylablib is guarded
# there), so these are safe even when running camera-only.
from stage import TRAVEL_MM as STAGE_MAX_MM, resolve_target

# Arrow-key codes vary by OS/GUI backend; match against all known up/down values
# (macOS Qt, GTK, Windows). w/s are provided as backend-independent alternates.
UP_KEYS   = {63232, 65362, 2490368, 82}
DOWN_KEYS = {63233, 65364, 2621440, 84}

# e/d move a finer step, FINE_RATIO times smaller than the w/s coarse step.
FINE_RATIO = 10

MAX_POINTS = 5000         # cap on stored (position, amplitude) samples

# Fine scan ('f'): stepped sweep around the CURRENT position — i.e. after the
# reference arm has been coarsely aligned by hand — measuring the patch contrast
# at each step. Ends by saving a contrast-vs-position plot and moving the stage
# to the coherence peak.
FINE_RANGE_MM = 0.5           # half-width of the sweep (+/- this around the current position)
FINE_STEP_MM = 0.010          # 10 um
FINE_DWELL_FRAMES = 4         # frames averaged at each step (stage parked, so this is free)
FINE_MIN_STEPS = 5            # compute_fwhm needs >= 5 points to find an envelope
FINE_STALL_FRAMES = 150       # ~5 s: give up on a target the stage never quite reaches

# ── Theme ─────────────────────────────────────────────────────────────────────
# Dark-surface palette validated for CVD separation and contrast (dataviz
# reference palette, dark column). cv2 wants BGR; Pillow wants RGB.
def _bgr(hexcode):
    r, g, b = (int(hexcode[i:i + 2], 16) for i in (1, 3, 5))
    return (b, g, r)

SURFACE     = _bgr("#1a1a19")   # window background
SURFACE_2   = _bgr("#222221")   # raised surfaces (status bar, tiles)
GRID        = _bgr("#33332f")   # plot grid / outlines
SERIES      = _bgr("#3987e5")   # amplitude curve (blue)
ACCENT      = _bgr("#9085e9")   # FWHM / peak annotations (violet)
LIVE        = _bgr("#199e70")   # current-position marker (aqua)
TEXT_1      = _bgr("#ffffff")   # primary text
TEXT_2      = _bgr("#c3c2b7")   # secondary text
TEXT_MUTED  = _bgr("#807f75")   # axis labels, hints

# Right-hand panel layout (pixels).
PLOT_W    = 520
PB_X0     = 64                  # plot box left edge inside the panel
PB_X1     = PLOT_W - 28         # plot box right edge
HEADER_H  = 58                  # title block above the tiles
TILES_H   = 84                  # stat-tile row height
STATUS_H  = 52                  # bottom status bar (full canvas width)
INSET_H   = 240                 # magnified-patch section, between tiles and plot
MIN_PLOT_H = 140                # drop the inset rather than squeeze the plot below this


class Text:
    """Antialiased text via Pillow with a geometric system font.

    Prefers Avenir Next / Avenir, then other clean faces. Draw calls are queued
    and rendered in ONE numpy->PIL->numpy round trip per frame (`flush`), which
    keeps the cost to a few ms. Falls back to cv2.putText if Pillow or every
    candidate font is missing.
    """

    # System font collections tried when no local font is found.
    CANDIDATES = [
        "/System/Library/Fonts/Avenir Next.ttc",
        "/System/Library/Fonts/Avenir.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Supplemental/Futura.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    WEIGHTS = {  # weight name -> acceptable face styles, best first
        "regular": ["Regular", "Book", "Roman", "Medium"],
        "medium":  ["Medium", "Demi Bold", "Regular"],
        "demi":    ["Demi Bold", "Bold", "Medium", "Heavy"],
    }
    # Filename keywords for per-weight font FILES (e.g. Ageo ships one
    # .otf per weight) in the local fonts/ directory, best first.
    FILE_WEIGHTS = {
        "regular": ["regular", "book", "roman"],
        "medium":  ["medium", "semibold", "regular"],
        "demi":    ["semibold", "demibold", "demi", "bold", "heavy", "medium"],
    }

    def __init__(self):
        self.ok = False
        self.queue = []
        self.cache = {}
        self._masks = {}   # (string, size, weight) -> (alpha mask, ascent)
        # Coordinates given to put() are relative to this origin — set it to a
        # sub-surface's top-left (panel, status bar) before queueing its text,
        # since everything is flushed once on the composed canvas.
        self.origin = (0, 0)
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image, self.ImageDraw, self.ImageFont = Image, ImageDraw, ImageFont
        except ImportError:
            print("Pillow not found — falling back to OpenCV fonts.")
            return
        self.faces = self._find_faces()
        if self.faces:
            self.ok = True
            path, idx = self.faces["regular"]
            fam = self.ImageFont.truetype(path, 12, index=idx).getname()[0]
            print(f"UI font: {fam}")

    def _find_faces(self):
        """Pick (path, index) per weight: local fonts/ dir first, then system."""
        return self._find_local_faces() or self._find_system_faces()

    def _find_local_faces(self):
        """Use font files from the project's font/ (or fonts/) directory.

        Searched recursively (e.g. font/Ageo/Ageo-Regular.ttf). Families that
        ship one file per weight (like Ageo) are matched by filename keywords;
        'ageo' files win over other local fonts. Only FreeType formats are
        considered — the .woff/.eot webfont copies are skipped.
        """
        from pathlib import Path
        here = Path(__file__).resolve().parent
        files = []
        for d in ("font", "fonts"):
            files += sorted(p for p in (here / d).rglob("*")
                            if p.suffix.lower() in (".otf", ".ttf", ".ttc"))
        if not files:
            return None
        # Prefer Ageo if present, else whichever family sorts first.
        ageo = [p for p in files if "ageo" in p.name.lower()]
        pool = ageo or files
        # Drop italic cuts — this UI only uses uprights.
        upright = [p for p in pool if "italic" not in p.name.lower()] or pool

        def pick(weight):
            for kw in self.FILE_WEIGHTS[weight]:
                for p in upright:
                    if kw in p.name.lower():
                        return p
            # No keyword hit (e.g. a single "Ageo.otf") — use the first file.
            return upright[0]

        faces = {}
        for weight in self.WEIGHTS:
            p = pick(weight)
            try:
                self.ImageFont.truetype(str(p), 12)   # verify it loads
            except Exception as e:
                print(f"Could not load {p.name} ({e}); using system fonts.")
                return None
            faces[weight] = (str(p), 0)
        return faces

    def _find_system_faces(self):
        """Pick (path, index) per weight from the first installed candidate."""
        import os
        for path in self.CANDIDATES:
            if not os.path.exists(path):
                continue
            styles = {}
            for idx in range(18):   # enumerate faces in the collection
                try:
                    name = self.ImageFont.truetype(path, 12, index=idx).getname()[1]
                    styles.setdefault(name, idx)
                except Exception:
                    break
            if not styles:
                continue
            faces = {}
            for weight, wanted in self.WEIGHTS.items():
                idx = next((styles[s] for s in wanted if s in styles), 0)
                faces[weight] = (path, idx)
            return faces
        return None

    def _font(self, size, weight):
        key = (size, weight)
        if key not in self.cache:
            path, idx = self.faces[weight]
            self.cache[key] = self.ImageFont.truetype(path, size, index=idx)
        return self.cache[key]

    def put(self, xy, s, size=13, color=TEXT_2, weight="regular", anchor="la"):
        """Queue a string; `anchor` is a PIL text anchor (la, ra, ma, ...).

        Coordinates stay origin-relative; flush() groups by origin and only
        round-trips those sub-surfaces through PIL.
        """
        self.queue.append((self.origin, xy, s, size, color, weight, anchor))

    def width(self, s, size, weight="regular"):
        """Rendered width of `s` in pixels (estimate under the cv2 fallback)."""
        if not self.ok:
            return len(s) * size * 0.55
        return self._font(size, weight).getlength(s)

    def _mask(self, s, size, weight):
        """Rasterize `s` ONCE into a cached grayscale alpha mask.

        PIL text rendering costs >1 ms per string on this Pillow build, which
        at ~20 strings/frame would eat the whole 33 ms frame budget. Labels
        and most values repeat frame-to-frame, so each unique string is
        rendered to a small (H, W) alpha sprite once and numpy-blended into
        the canvas per frame (~µs each). Returns (mask float32 0..1, ascent).
        """
        key = (s, size, weight)
        hit = self._masks.get(key)
        if hit is not None:
            return hit
        font = self._font(size, weight)
        ascent, descent = font.getmetrics()
        w = max(int(font.getlength(s)) + 2, 1)
        img = self.Image.new("L", (w, ascent + descent + 2), 0)
        self.ImageDraw.Draw(img).text((0, 0), s, font=font, fill=255)
        mask = np.asarray(img, dtype=np.float32) / 255.0
        if len(self._masks) > 2048:      # bound the cache (fast-changing values)
            self._masks.clear()
        self._masks[key] = (mask, ascent)
        return self._masks[key]

    def flush(self, canvas):
        """Blend all queued strings onto the BGR canvas; returns the canvas.

        Each string is a cached alpha sprite blitted onto its small canvas
        region — no full-canvas or region-wide PIL round trips.
        """
        if not self.queue:
            return canvas
        if not self.ok:                      # Hershey fallback
            for (ox, oy), (x, y), s, size, color, weight, anchor in self.queue:
                x, y = x + ox, y + oy
                if anchor.startswith("r"):
                    x -= int(len(s) * size * 0.55)
                elif anchor.startswith("m"):
                    x -= int(len(s) * size * 0.28)
                cv2.putText(canvas, s, (int(x), int(y + size)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 26.0, color, 1,
                            cv2.LINE_AA)
            self.queue.clear()
            return canvas
        ch, cw = canvas.shape[:2]
        for (ox, oy), (x, y), s, size, color, weight, anchor in self.queue:
            mask, ascent = self._mask(s, size, weight)
            mh, mw = mask.shape
            x, y = x + ox, y + oy
            # PIL-style anchors: 1st char l/m/r horizontal, 2nd a/m/s vertical.
            if anchor[0] == "m":
                x -= mw / 2
            elif anchor[0] == "r":
                x -= mw
            if anchor[1] == "m":
                y -= mh / 2
            elif anchor[1] == "s":
                y -= ascent
            x, y = int(x), int(y)
            # Clip the sprite to the canvas.
            sx0, sy0 = max(-x, 0), max(-y, 0)
            sx1, sy1 = min(mw, cw - x), min(mh, ch - y)
            if sx1 <= sx0 or sy1 <= sy0:
                continue
            a = mask[sy0:sy1, sx0:sx1, None]
            region = canvas[y + sy0:y + sy1, x + sx0:x + sx1]
            region[:] = (region * (1.0 - a)
                         + np.asarray(color, np.float32) * a).astype(np.uint8)
        self.queue.clear()
        return canvas


_GAMMA_LUTS = {}   # (dtype bits, maxv, gamma) -> uint8 lookup table


def to_display_8bit(frame, gamma):
    """Convert a raw frame to a gamma-corrected 8-bit image for human viewing.

    The data is normalized to [0, 1] by its bit depth, then encoded with the
    standard display gamma (out = in**(1/gamma); gamma > 1 brightens the shadows
    so a dark linear scene becomes visible), then scaled to 8-bit. Gamma is
    applied on the full-bit-depth data, before quantizing, so shadow detail
    isn't lost to an early bit-shift.

    Integer frames go through a precomputed per-value lookup table: np.power
    over a full-resolution frame costs ~50 ms, the LUT gather ~5 ms, which is
    what makes an unbinned live preview feasible.
    """
    def lut_for(nvals, maxv):
        key = (nvals, maxv, gamma)
        lut = _GAMMA_LUTS.get(key)
        if lut is None:
            norm = np.arange(nvals, dtype=np.float32) / maxv
            if gamma > 0 and gamma != 1.0:
                norm = np.power(norm, 1.0 / gamma, dtype=np.float32)
            lut = (norm * 255.0).clip(0, 255).astype(np.uint8)
            _GAMMA_LUTS[key] = lut
        return lut

    if frame.dtype == np.uint8:
        return cv2.LUT(frame, lut_for(256, 255.0))
    if frame.dtype == np.uint16:
        # 12-bit sensors store values <= 4095 in a 16-bit container.
        maxv = 4095.0 if (frame.size and int(frame.max()) <= 4095) else 65535.0
        return lut_for(65536, maxv)[frame]
    mx = float(frame.max()) if frame.size else 0.0
    norm = frame.astype(np.float32) / (mx if mx > 0 else 1.0)
    if gamma > 0 and gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma, dtype=np.float32)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


def get_screen_size():
    """Return (width, height) of the primary screen, with a sane fallback.

    Runs tkinter in a separate process: initializing Tk in this process would
    spin up its own macOS NSApplication, which collides with OpenCV's Qt
    backend and crashes the GUI (NSException / abort trap).
    """
    import subprocess
    import sys
    try:
        code = (
            "import tkinter; r=tkinter.Tk(); r.withdraw();"
            "print(r.winfo_screenwidth(), r.winfo_screenheight())"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        w, h = (int(x) for x in out.split())
        return w, h
    except Exception:
        return 1280, 720


def connect_stage():
    """Try to connect a Thorlabs stage. Return the stage or None if unavailable."""
    try:
        from stage import ThorlabsStage
        stage = ThorlabsStage(units="mm")
        stage.connect()
        print("Stage connected — arrow keys (or w/s) move it.")
        # Only home if the controller hasn't been homed since power-on: a
        # restart mid-session shouldn't drive the stage back to 0 and cost a
        # homing cycle when it already has a valid datum.
        if not stage.home_if_needed():
            print(f"Stage already homed — keeping position "
                  f"{stage.position:.4f} mm.")
        return stage
    except Exception as e:
        print(f"No stage found ({e}); running camera-only.")
        return None


class PatchSelector:
    """Two-click rectangular patch selection in *frame* (not display) coordinates.

    First click anchors a corner; the rectangle then follows the mouse until a
    second click finalizes it. `rect` is (x0, y0, x1, y1) with x0<x1, y0<y1,
    or None while nothing is finalized.
    """

    def __init__(self):
        self.anchor = None      # first click, frame coords
        self.hover = None       # current mouse position, frame coords
        self.rect = None        # finalized (x0, y0, x1, y1)

    def click(self, x, y):
        if self.anchor is None:
            self.anchor = (x, y)
            self.rect = None
        else:
            self.rect = self._ordered(self.anchor, (x, y))
            self.anchor = None
        return self.rect

    def move(self, x, y):
        self.hover = (x, y)

    def clear(self):
        self.anchor = None
        self.rect = None

    def preview_rect(self):
        """Rectangle to draw: the live anchor→mouse box, or the finalized one."""
        if self.anchor is not None and self.hover is not None:
            return self._ordered(self.anchor, self.hover)
        return self.rect

    @staticmethod
    def _ordered(p0, p1):
        x0, x1 = sorted((p0[0], p1[0]))
        y0, y1 = sorted((p0[1], p1[1]))
        # Enforce a minimum 2x2 patch so the amplitude is well-defined.
        return (x0, y0, max(x1, x0 + 2), max(y1, y0 + 2))


def clip_rect(rect, shape):
    """Clip a frame-coordinate (x0, y0, x1, y1) to the image bounds.

    Returns the clipped rect, or None if it lands outside the frame or is
    smaller than the 2x2 the amplitude measure needs.
    """
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    fh, fw = shape[:2]
    x0, x1 = (int(v) for v in np.clip((x0, x1), 0, fw))
    y0, y1 = (int(v) for v in np.clip((y0, y1), 0, fh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


class PatchAmplitude:
    """Position-windowed interference amplitude of a patch.

    Loose streaming version of compute_mean_diff (computeMeanDiff.m) with
    avg_type='local', but the "stack" is indexed by *stage position* instead of
    frame number: because the stage moves irregularly (and sometimes not at
    all), a fixed-N temporal window would mix wildly different z-spacings. So
    we buffer (position, patch) pairs and the DC estimate for the newest frame
    is the mean of all buffered patches within ±window_mm of its position;
    amplitude = mean(|patch - that local mean|) — the same measure
    get_int_amplitude averages spatially.
    """

    def __init__(self, window_mm, max_frames=400):
        self.window_mm = window_mm
        self.buf = deque(maxlen=max_frames)   # (position_mm, patch) pairs
        # Running sum over ALL buffered patches, maintained incrementally
        # (add on append, subtract on eviction). When every buffered position
        # is inside the window — the common case: stage parked, or a sweep
        # narrower than the window — the DC mean is this sum / len(buf),
        # O(patch) per frame instead of restacking up to 400 patches (which
        # for a large patch is tens of MB of allocation per frame at 30 fps).
        self._sum = None

    def reset(self):
        self.buf.clear()
        self._sum = None

    def update(self, patch, pos):
        """Add a (pos, patch) sample; return the amplitude at `pos`, or None.

        `pos` may be any monotone-ish coordinate — stage mm normally, or a
        frame index (with window_mm reinterpreted in frames) when no stage.
        """
        # float32: half the memory of float64 and ample precision for a mean
        # of <=12-bit camera data over a <=400-deep buffer.
        patch = patch.astype(np.float32)
        # A patch-size change (new selection) invalidates the buffer.
        if self.buf and self.buf[0][1].shape != patch.shape:
            self.reset()
        if len(self.buf) == self.buf.maxlen:            # about to evict oldest
            self._sum -= self.buf[0][1]
        self.buf.append((pos, patch))
        if self._sum is None:
            self._sum = np.zeros_like(patch, dtype=np.float64)
        self._sum += patch

        if len(self.buf) < 2:
            return None
        # DC estimate: mean of the patches whose position falls in the window.
        # Positions are scanned cheaply (floats only); the patch arrays are
        # only restacked when part of the buffer falls outside the window.
        in_win = [abs(q - pos) <= self.window_mm for (q, _) in self.buf]
        n_in = sum(in_win)
        if n_in < 2:
            return None
        if n_in == len(self.buf):
            avg = self._sum / n_in                       # fast path, O(patch)
        else:
            avg = np.mean([p for (ok, (_, p)) in zip(in_win, self.buf) if ok],
                          axis=0)
        return float(np.mean(np.abs(patch - avg)))


def compute_fwhm(px, py):
    """Coherence length via full-width-half-max of the amplitude envelope.

    The floor (background amplitude away from the coherence peak) is
    subtracted first: half-max = floor + (peak - floor) / 2. Crossings on each
    side of the peak are linearly interpolated. Inputs must be sorted by
    position. Returns (fwhm, left_x, right_x, half_level, floor, peak_x) or
    None when the curve doesn't cross half-max on both sides of the peak.
    peak_x is the envelope centre — the midpoint of the two half-max crossings
    (more noise-robust than the argmax sample itself).
    """
    if len(px) < 5:
        return None
    floor = float(np.percentile(py, 10))       # robust background estimate
    ipk = int(np.argmax(py))
    half = floor + (py[ipk] - floor) / 2.0
    if py[ipk] - floor <= 0:
        return None

    def cross(i_from, step):
        """Walk from the peak until y drops below half; interpolate the x."""
        i = i_from
        while 0 <= i + step < len(px):
            j = i + step
            if py[j] < half:
                t = (half - py[i]) / (py[j] - py[i])
                return float(px[i] + t * (px[j] - px[i]))
            i = j
        return None

    left = cross(ipk, -1)
    right = cross(ipk, +1)
    if left is None or right is None:
        return None
    return right - left, left, right, half, floor, (left + right) / 2.0


def save_contrast_plot(px, py, peak_pos, fwhm_res, step_mm, cam):
    """Save the fine scan as a contrast-vs-position PNG (+ a CSV of the data).

    Matplotlib MUST be pinned to the non-interactive Agg backend before pyplot
    is imported: this process already hosts an OpenCV Qt GUI, and a default
    import picks the macosx backend, which spins up a second NSApplication in
    the same process — the toolkit collision that get_screen_size() documents
    (NSException / abort trap). Agg is pure raster: no toolkit, no event loop.
    The import is lazy so a plain live-view session never pays for it.

    Returns the Path of the PNG.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)          # must precede the pyplot import
    import matplotlib.pyplot as plt
    from pathlib import Path

    save_dir = Path(getattr(cam, "save_dir", "live_view_captures"))
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / cam.timestamped_filename(prefix="fine_scan", ext="png")

    def hexc(bgr):                             # theme colors are BGR, pyplot wants RGB
        b, g, r = bgr
        return f"#{r:02x}{g:02x}{b:02x}"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(px, py, "-o", ms=3, lw=1.2, color=hexc(SERIES),
            label="contrast  mean |I − DC|")
    ax.axvline(peak_pos, color=hexc(ACCENT), ls="--", lw=1.2,
               label=f"peak {peak_pos:.4f} mm")
    title = (f"Fine scan · {len(px)} steps of {step_mm * 1000:.0f} µm "
             f"({px[0]:.3f}–{px[-1]:.3f} mm)")
    if fwhm_res is not None:
        width, lx, rx, half, floor, _ = fwhm_res
        ax.hlines(half, lx, rx, color=hexc(ACCENT), lw=1.2)
        ax.axhline(floor, color="#999999", ls=":", lw=0.8, label="floor")
        ax.annotate(f"FWHM = {width * 1000:.1f} µm", ((lx + rx) / 2, half),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    color=hexc(ACCENT))
        title += f"  ·  coherence FWHM {width * 1000:.1f} µm"
    ax.set_xlabel("stage position (mm)")
    ax.set_ylabel("contrast  mean |I − DC|  (counts)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    csv_path = path.with_suffix(".csv")
    csv_path.write_text("position_mm,contrast\n"
                        + "".join(f"{x:.6f},{y:.6f}\n" for x, y in zip(px, py)))
    print(f"Saved: {path}\nSaved: {csv_path}")
    return path


def _dashed_line(img, p0, p1, color, dash=5, gap=4, thickness=1):
    """Draw a dashed line segment (cv2 has no native dashes)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1:
        return
    d = (p1 - p0) / length
    t = 0.0
    while t < length:
        a = p0 + d * t
        b = p0 + d * min(t + dash, length)
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)),
                 color, thickness, cv2.LINE_AA)
        t += dash + gap


def _tile(panel, txt, x, y, w, h, label, value, unit, color=TEXT_1):
    """One stat tile: small caps label on top, big value + unit below.

    Value and unit share a baseline ("ls" anchors); the unit is placed after
    the value's measured width so they never collide.
    """
    cv2.rectangle(panel, (x, y), (x + w, y + h), SURFACE_2, -1)
    txt.put((x + 12, y + 10), label.upper(), 10, TEXT_MUTED, "medium")
    txt.put((x + 12, y + 54), value, 24, color, "demi", anchor="ls")
    if unit and value != "—":
        vw = txt.width(value, 24, "demi")
        txt.put((x + 12 + int(vw) + 6, y + 54), unit, 11, TEXT_MUTED, "medium",
                anchor="ls")


def stretch_patch(crop, gamma, max_side=600):
    """Percentile-stretch a RAW crop to 8-bit so faint fringes are visible.

    The 2nd–98th percentile of the crop is mapped to 0–255, then the display
    gamma is applied. Working from the RAW crop (not the already-gamma'd 8-bit
    display image) matters: stretching after quantization can only rescale the
    few levels a dim patch survived with, while stretching the full-depth data
    recovers detail that was there all along — a fringe pattern spanning one
    grey level in the normal view comes back as a full-range image.

    Oversized crops are decimated first. The normalize + gamma below is
    per-pixel, and on a 1200x1200 selection it costs ~20 ms — most of the 33 ms
    frame budget — to compute detail that cannot be shown, since the inset box
    is only ~464 px wide and would shrink it anyway.
    """
    step = max(1, max(crop.shape) // max_side)
    if step > 1:
        crop = crop[::step, ::step]
    lo, hi = np.percentile(crop, (2, 98))
    if hi <= lo:              # percentiles coincide (flat, or constant + outlier)
        lo, hi = float(crop.min()), float(crop.max())
        if hi <= lo:                               # genuinely flat
            return np.zeros(crop.shape, np.uint8)
    norm = ((crop.astype(np.float32) - lo) / (hi - lo)).clip(0.0, 1.0)
    if gamma > 0 and gamma != 1.0:
        norm = np.power(norm, 1.0 / gamma, dtype=np.float32)
    return (norm * 255.0).astype(np.uint8)


def fit_scale(cw, ch, bw, bh):
    """Scale factor that fits a cw x ch crop inside a bw x bh box, aspect preserved."""
    if cw <= 0 or ch <= 0:
        return 1.0
    return min(bw / cw, bh / ch)


def render_inset(panel, txt, crop8, x, y, w, h, auto, patch_wh):
    """Draw the magnified patch into the panel's inset box at (x, y, w, h).

    `patch_wh` is the TRUE selected size in sensor pixels — crop8 may have been
    decimated by stretch_patch, so the label and the magnification factor have
    to come from the real selection, not from the array being drawn.
    """
    txt.put((x, y), "PATCH", 10, TEXT_MUTED, "medium")
    # The stretch state lives on the label, not in the status bar: the bar's key
    # hint is already the widest thing in it, and the hint belongs next to the
    # thing it affects. It must always be visible when auto is on — a stretched
    # view exaggerates contrast and hides how close the patch is to saturation.
    txt.put((x + w, y), "i: auto-contrast" if auto else "i: true levels",
            10, LIVE if auto else TEXT_MUTED, "medium", anchor="ra")

    box_y = y + 18                       # image area, below the label row
    box_h = h - 18
    ch, cw = crop8.shape[:2]
    s = fit_scale(cw, ch, w, box_h)
    dw, dh = max(int(cw * s), 1), max(int(ch * s), 1)
    # Magnifying: NEAREST shows true sensor pixels (blocky), which is the point —
    # a smooth interpolation would invent detail and blur the fringes away.
    interp = cv2.INTER_NEAREST if s >= 1.0 else cv2.INTER_AREA
    big = cv2.resize(crop8, (dw, dh), interpolation=interp)
    if big.ndim == 2:
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

    ox = x + (w - dw) // 2               # centre it in the box
    oy = box_y + (box_h - dh) // 2
    panel[oy:oy + dh, ox:ox + dw] = big
    # Same blue as the finalized patch rectangle on the image, so the inset and
    # the box it came from read as the same object.
    cv2.rectangle(panel, (ox - 1, oy - 1), (ox + dw, oy + dh), SERIES, 1)

    # Size and zoom are quoted against the true selection, not the (possibly
    # decimated) array above.
    pw, ph = patch_wh
    txt.put((x, y + h - 2), f"{pw}×{ph} px  ·  {dw / max(pw, 1):.1f}×", 10,
            TEXT_MUTED, anchor="ls")


def render_panel(txt, points, height, live_val=None, current_pos=None,
                 view=None, has_stage=True, cache=None, version=None,
                 inset=None, inset_auto=False, inset_px=None):
    """Render the right-hand panel: header, stat tiles, magnified patch, plot.

    `points` is a sequence of (position, amplitude), one value per position.
    `inset` is an 8-bit crop of the selected patch (already stretched or not),
    drawn magnified between the tiles and the plot; None = no patch selected.
    `view` is an optional (xmin, xmax) zoom range; None = auto-fit all data.
    `cache`/`version`: samples change far less often than frames render, so
    the caller passes a dict and a counter it bumps on every samples mutation;
    the sorted arrays + FWHM are recomputed only when the counter moved.
    Returns (panel, (xmin, xmax)) — the x-range actually drawn, so mouse events
    on the panel can be mapped back to data coordinates for zoom/pan.
    Geometry (grid, curve, markers) is cv2 with LINE_AA; all text is queued on
    `txt` and rendered by the caller's single flush.
    """
    panel = np.full((height, PLOT_W, 3), SURFACE, np.uint8)
    unit = "mm" if has_stage else "frame"

    # ── Header ──
    txt.put((28, 16), "COHERENCE SCAN", 17, TEXT_1, "demi")
    txt.put((28, 38), f"patch amplitude vs stage position ({unit})",
            11, TEXT_MUTED)

    # ── FWHM / peak (computed on ALL data, independent of zoom) ──
    if cache is None:
        cache = {}
    if cache.get("version") != version or version is None:
        if len(points) >= 2:
            pts = np.asarray(points, dtype=np.float64)
            order = np.argsort(pts[:, 0])
            px, py = pts[order, 0], pts[order, 1]
            cache.update(version=version, px=px, py=py,
                         fwhm=compute_fwhm(px, py))
        else:
            cache.update(version=version, px=None, py=None, fwhm=None)
    px, py, fwhm = cache["px"], cache["py"], cache["fwhm"]

    # ── Stat tiles ──
    ty = HEADER_H
    tw = (PLOT_W - 28 * 2 - 12 * 2) // 3
    if fwhm is not None:
        width, lx, rx, half, floor, peak_x = fwhm
        fwhm_s = f"{width * 1000:.1f}" if has_stage else f"{width:.1f}"
        peak_s = f"{peak_x:.4f}" if has_stage else f"{peak_x:.0f}"
        fu, pu = ("um", "mm") if has_stage else ("", "")
    else:
        fwhm_s, peak_s, fu, pu = "—", "—", "", ""
    live_s = f"{live_val:.4g}" if live_val is not None else "—"
    _tile(panel, txt, 28, ty, tw, 62, "coherence fwhm", fwhm_s, fu, ACCENT)
    _tile(panel, txt, 28 + tw + 12, ty, tw, 62, "peak", peak_s, pu, ACCENT)
    _tile(panel, txt, 28 + 2 * (tw + 12), ty, tw, 62, "amplitude", live_s, "", SERIES)

    # ── Magnified patch ──
    # Skipped entirely on a short panel rather than crushing the plot below a
    # usable height (a small sensor or a small window makes the panel short).
    bot = height - 34
    inset_h = 0
    if inset is not None and (bot - (HEADER_H + TILES_H + INSET_H)) >= MIN_PLOT_H:
        inset_h = INSET_H
        wh = inset_px or (inset.shape[1], inset.shape[0])
        render_inset(panel, txt, inset, 28, HEADER_H + TILES_H + 6,
                     PLOT_W - 28 * 2, INSET_H - 18, inset_auto, wh)

    # ── Plot box ──
    top = HEADER_H + TILES_H + inset_h
    if bot - top < 60:
        return panel, None

    if px is None:
        msg = "select a patch — two clicks on the image" if not points \
              else "move the stage to build the curve"
        txt.put((PLOT_W // 2, (top + bot) // 2), msg, 12, TEXT_MUTED,
                anchor="mm")
        return panel, None

    # x-range: the zoom view if set, else auto-fit all data.
    if view is not None:
        xmin, xmax = view
    else:
        xmin, xmax = float(px.min()), float(px.max())
    if xmax - xmin < 1e-9:
        xmax = xmin + 1e-9

    # y-range: fit the points visible in the current x-range so zooming into
    # a region rescales vertically too. Starts at the visible minimum (not 0)
    # so the peak spans most of the plot height.
    vis = (px >= xmin) & (px <= xmax)
    vy = py[vis] if vis.any() else py
    ymin, ymax = float(vy.min()), float(vy.max())
    pad = (ymax - ymin) * 0.07
    ymin, ymax = ymin - pad, ymax + pad
    if ymax - ymin < 1e-12:
        ymax = ymin + 1e-12

    def to_px(x, y):
        return (int(PB_X0 + (x - xmin) / (xmax - xmin) * (PB_X1 - PB_X0)),
                int(bot - (y - ymin) / (ymax - ymin) * (bot - top)))

    # Recessive grid: 4 horizontal lines + y labels; box only on the bottom.
    for i in range(5):
        gy = top + (bot - top) * i // 4
        cv2.line(panel, (PB_X0, gy), (PB_X1, gy), GRID, 1, cv2.LINE_AA)
        yv = ymax - (ymax - ymin) * i / 4
        txt.put((PB_X0 - 8, gy), f"{yv:.3g}", 10, TEXT_MUTED, anchor="rm")

    # Floor line (may sit below a zoomed-in y-range).
    if fwhm is not None:
        fl = to_px(xmin, floor)[1]
        if top <= fl <= bot:
            _dashed_line(panel, (PB_X0, fl), (PB_X1, fl), GRID, 6, 5)

    # Curve: 2px antialiased polyline through the visible points, mapped to
    # pixels with one vectorized affine transform (px is already sorted).
    seg = np.empty((int(vis.sum()), 2), np.int32)
    seg[:, 0] = PB_X0 + (px[vis] - xmin) / (xmax - xmin) * (PB_X1 - PB_X0)
    seg[:, 1] = bot - (py[vis] - ymin) / (ymax - ymin) * (bot - top)
    # More samples than pixel columns is invisible but expensive to stroke —
    # collapse to one (column, mean y) vertex per column before drawing.
    if len(seg) > (PB_X1 - PB_X0):
        cols = seg[:, 0] - PB_X0
        counts = np.bincount(cols)
        ysum = np.bincount(cols, weights=seg[:, 1])
        keep = counts > 0
        seg = np.column_stack([np.nonzero(keep)[0] + PB_X0,
                               (ysum[keep] / counts[keep])]).astype(np.int32)
    if len(seg) >= 2:
        cv2.polylines(panel, [seg], False, SERIES, 2, cv2.LINE_AA)
    if len(seg) <= 300:   # dots add nothing on a dense curve
        for p in seg:
            cv2.circle(panel, tuple(p), 2, SERIES, -1, cv2.LINE_AA)

    # FWHM annotation: half-max segment with end ticks, dashed peak vertical.
    if fwhm is not None:
        pl, pr = to_px(lx, half), to_px(rx, half)
        cv2.line(panel, (max(pl[0], PB_X0), pl[1]), (min(pr[0], PB_X1), pr[1]),
                 ACCENT, 1, cv2.LINE_AA)
        for p in (pl, pr):
            if PB_X0 <= p[0] <= PB_X1:
                cv2.line(panel, (p[0], p[1] - 5), (p[0], p[1] + 5), ACCENT, 2,
                         cv2.LINE_AA)
        pk = to_px(peak_x, 0)[0]
        if PB_X0 <= pk <= PB_X1:
            _dashed_line(panel, (pk, top), (pk, bot), ACCENT, 4, 5)

    # Marker at the current stage position.
    if current_pos is not None:
        i = int(np.argmin(np.abs(px - current_pos)))
        p = to_px(px[i], py[i])
        if PB_X0 <= p[0] <= PB_X1:
            cv2.circle(panel, p, 5, LIVE, 2, cv2.LINE_AA)

    # x labels + zoom badge.
    txt.put((PB_X0, bot + 8), f"{xmin:.4f}", 10, TEXT_MUTED)
    txt.put((PB_X1, bot + 8), f"{xmax:.4f} {unit}", 10, TEXT_MUTED, anchor="ra")
    if view is not None:
        txt.put((PB_X1, top - 16), "ZOOMED · press a to autoscale", 10, LIVE,
                "medium", anchor="ra")
    txt.put(((PB_X0 + PB_X1) // 2, bot + 8), f"{len(points)} pts", 10,
            TEXT_MUTED, anchor="ma")

    return panel, (xmin, xmax)


def render_status(txt, width, exposure_us, gamma, fps, binning, stage_ok,
                  pos_mm, moving, coarse_mm, banner=None):
    """Full-width bottom status bar: camera stats left, stage centre, keys right.

    `banner` is an optional (text, color) that replaces the centre stage text —
    used for fine-scan progress and transient notices.
    """
    bar = np.full((STATUS_H, width, 3), SURFACE_2, np.uint8)
    cv2.line(bar, (0, 0), (width, 0), GRID, 1)
    cy = STATUS_H // 2

    # "gamma" spelled out: Ageo has no Greek glyphs, so a γ would drop out.
    # Two lines on the left: the values, and how to change them (the exposure
    # hint lives here, next to the number it adjusts).
    cam_s = f"{int(exposure_us)} µs    gamma {gamma:g}    {fps:0.1f} fps    bin {binning}×"
    txt.put((20, cy - 9), cam_s, 16, TEXT_1, "demi", anchor="lm")
    txt.put((20, cy + 13), "press  −  to dim   /   +  to brighten  (exposure)",
            12, TEXT_MUTED, "medium", anchor="lm")

    if banner is not None:
        stage_s, color = banner
    elif stage_ok:
        pos_s = f"{pos_mm:.4f} mm" if pos_mm is not None else "—"
        stage_s = f"stage {pos_s}    step {coarse_mm:g} / {coarse_mm / FINE_RATIO:g} mm"
        color = LIVE if moving else TEXT_1
        if moving:
            stage_s += "    MOVING"
    else:
        stage_s, color = "stage not found — camera only", TEXT_MUTED
    txt.put((width // 2, cy), stage_s, 16, color, "demi", anchor="mm")

    txt.put((width - 20, cy), "hold w/e sweep   s/d back   [ ] step   m go to   f fine scan   c clear   a fit   q quit",
            14, TEXT_2, "medium", anchor="rm")
    return bar


def main():
    parser = argparse.ArgumentParser(
        description="Live camera viewer with stage control and coherence panel.")
    parser.add_argument("--step", type=float, default=1.0,
                        help="coarse stage step in mm (w/s or arrows); e/d "
                             f"moves 1/{FINE_RATIO} of this (default 1.0)")
    parser.add_argument("--binning", type=int, default=1,
                        help="on-sensor binning factor for the preview "
                             "(default 1 = full resolution; try 2 or 4 if the "
                             "preview lags)")
    parser.add_argument("--exposure", type=float, default=None,
                        help="initial exposure in microseconds, non-binned-"
                             "equivalent (default: your last session's value "
                             "from last_exposure.json, else 10000)")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="display gamma; >1 brightens dark scenes (default 2.2, 1.0 = linear)")
    parser.add_argument("--window-mm", type=float, default=0.2,
                        help="half-width in mm of the positional window used for "
                             "the DC (local mean) estimate; patches captured "
                             "within +/- this distance of the current position "
                             "are averaged (default 0.2). With no stage it is "
                             "interpreted in frames.")
    parser.add_argument("--fine-range-mm", type=float, default=FINE_RANGE_MM,
                        help="half-width in mm of the 'f' fine scan around the "
                             f"current position (default {FINE_RANGE_MM:g}, i.e. "
                             "a 1 mm sweep)")
    parser.add_argument("--fine-step-mm", type=float, default=FINE_STEP_MM,
                        help="step size in mm of the 'f' fine scan "
                             f"(default {FINE_STEP_MM:g} = "
                             f"{FINE_STEP_MM * 1000:g} um)")
    args = parser.parse_args()
    gamma = args.gamma
    fine_range, fine_step = args.fine_range_mm, args.fine_step_mm

    # Default exposure: pick up where the last session left off (the value is
    # saved to last_exposure.json on every adjustment).
    exposure_init = args.exposure
    if exposure_init is None:
        from exposure_store import load_exposure
        stored = load_exposure()
        if stored is not None:
            exposure_init, age = stored
            print(f"Restoring exposure from your last session: "
                  f"{exposure_init:g} µs  (saved {age})")
        else:
            exposure_init = 10000

    txt = Text()

    print("Initializing camera...")
    from camera import Camera
    cam = Camera(exposure_us=exposure_init, gain_db=0.0, save_dir="live_view_captures")
    cam.connect()

    stage = connect_stage()
    coarse_mm = args.step   # w/s or arrows; e/d moves coarse_mm / FINE_RATIO

    exposure_us = cam.exposure_us
    exposure_step = 500  # microseconds
    frame_time_target = 1.0 / 30.0  # 30 FPS

    # Bin on-sensor for the live preview. NxN binning sums N**2 pixels, so the
    # image is ~N**2 brighter; we divide the actual exposure by binning**2 so the
    # exposure *shown* equals the non-binned-equivalent value usable elsewhere.
    live_binning = args.binning
    try:
        cam.set_binning(live_binning)
    except Exception as e:
        print(f"Binning unavailable, running at full resolution: {e}")
        live_binning = 1

    def apply_exposure(shown_us):
        """Set the camera exposure compensated for binning brightness gain.

        Also recorded to last_exposure.json so oct_scan.py can reuse the value
        — the IDS driver resets exposure on re-init, so the sensor itself
        can't carry it between programs.
        """
        try:
            cam.set_exposure(max(shown_us / (live_binning ** 2), 1))
        except Exception as e:
            print(f"Could not set exposure: {e}")
            return
        from exposure_store import save_exposure
        save_exposure(shown_us)

    apply_exposure(exposure_us)

    def move_stage(direction, fine=False):
        """Move the stage (non-blocking so the preview keeps running)."""
        if stage is None:
            return
        dist = coarse_mm / FINE_RATIO if fine else coarse_mm
        try:
            stage.move_by(direction * dist, wait=False)
        except Exception as e:
            print(f"Stage move failed: {e}")

    # Size the window to fit the screen (leave a margin for window chrome and
    # the panel on the right / status bar below).
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9) - PLOT_W
    max_h = int(screen_h * 0.9) - STATUS_H

    window_name = "Visualizer"
    # WINDOW_GUI_NORMAL disables OpenCV's Qt toolbar. The toolbar's icon engine
    # (QAppleIconEngine) crashes on Qt 6.11 + macOS 12 with a doesNotRecognizeSelector
    # NSException, so we must not create it. WINDOW_NORMAL keeps the window resizable.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    window_sized = False

    # Patch selection + amplitude state. Mouse coords arrive in canvas space
    # (image scaled by `scale`, panel on the right, status bar below);
    # `disp_state` holds the current scale and image size so the callback can
    # map back to frame coordinates and route panel events to zoom/pan.
    selector = PatchSelector()
    # With no stage the x-coordinate is a frame index, so a sub-1 window in
    # "mm" would never catch neighbors — widen it to a frame count instead.
    window = args.window_mm if stage is not None else max(args.window_mm, 10)
    amp = PatchAmplitude(window)
    # One value per position: keyed by position rounded to 0.1 µm, so sitting
    # at (or revisiting) a position overwrites its value instead of adding points.
    samples = {}                         # pos_key -> (position mm, amplitude)
    # Bumped on every samples mutation; render_panel caches the sorted
    # arrays + FWHM against it so they aren't recomputed every frame.
    samples_version = 0
    panel_cache = {}
    # Fine scan ('f'): non-blocking stepped sweep around the current position,
    # then save the contrast plot and move to the coherence peak. Runs as a
    # state machine inside the main loop so the preview keeps streaming and f
    # cancels. See start_scan() for the state shape.
    scan = None
    # Typed position entry ('m'): None = normal keys; a string = what's been
    # typed so far (a leading +/- makes it a relative move).
    entry = None
    # 'i': percentile-stretch the magnified patch inset so faint fringes show.
    # Off by default — the inset then matches the live view exactly.
    inset_auto = False
    notice = None        # transient status-bar message: (text, expires_at)
    # plot_view["range"]: (xmin, xmax) zoom of the plot, None = auto-fit.
    # plot_view["drawn"]: the range actually rendered last frame (needed to map
    # panel pixels back to data coordinates for wheel-zoom / drag-pan).
    plot_view = {"range": None, "drawn": None, "drag": None}
    disp_state = {"scale": 1.0, "img_w": 0, "img_h": 0}

    def start_scan(targets, step):
        """Begin a stepped sweep. Returns the state, or None.

        The plot is NOT cleared here: it is reset when the stage ARRIVES at
        the first target (see the state machine), so the transit from the
        current position doesn't repopulate the panel with stray points
        before the scan proper starts.

        `tol` is stored per-scan and derived from THIS scan's step: the arrival
        test must not be scaled by the coarse jog step, which can be an order of
        magnitude larger than a fine step (every target would then read as
        "arrived" immediately and the sweep would record garbage).
        """
        st = {"targets": targets, "i": 0, "dwell": 0, "stall": 0, "step": step,
              "tol": max(step / 4, 5e-4),
              "cleared": False,       # plot reset happens on ARRIVAL at step 1
              "acc": [],              # live values seen during the current dwell
              "values": []}           # one (position, contrast) per completed step
        try:
            stage.move_to(targets[0], wait=False)
        except Exception as e:
            print(f"Fine scan start failed: {e}")
            return None
        return st

    def finish_scan(st, now):
        """End of sweep: pick the peak, move there, save the plot. Returns a notice."""
        if len(st["values"]) < 2:
            print("Fine scan: no samples collected.")
            return ("fine scan: no samples collected", now + 5)
        vals = sorted(st["values"])
        px = np.array([p for p, _ in vals])
        py = np.array([a for _, a in vals])

        # The envelope centre (midpoint of the half-max crossings) is more
        # robust than the argmax, which a single shot-noisy frame can win. It
        # only exists when the sweep actually brackets the peak; if it doesn't,
        # fall back to the best sample seen and say so.
        res = compute_fwhm(px, py)
        if res is not None:
            peak_pos, how = float(res[5]), "envelope centre"
        else:
            peak_pos, how = float(px[int(np.argmax(py))]), "argmax"
        peak_pos = float(np.clip(peak_pos, 0.0, STAGE_MAX_MM))

        # Issue the move BEFORE rendering the plot so the motion overlaps
        # matplotlib's first-import cost (a few hundred ms).
        try:
            stage.move_to(peak_pos, wait=False)
        except Exception as e:
            print(f"Fine scan: move to peak failed: {e}")
        msg = f"fine scan: peak at {peak_pos:.4f} mm ({how}) — moving there"
        print(f"Fine scan: peak at {peak_pos:.4f} mm via {how}; moving there.")
        if res is not None:
            print(f"Coherence FWHM: {res[0] * 1000:.1f} µm")

        try:
            path = save_contrast_plot(px, py, peak_pos, res, st["step"], cam)
            msg += f"  ·  saved {path.name}"
        except Exception as e:
            print(f"Could not save the contrast plot: {e}")
        return (msg, now + 8)

    def panel_to_data(panel_x):
        """Map an x pixel inside the plot panel to a data (stage) coordinate."""
        drawn = plot_view["drawn"]
        if drawn is None:
            return None
        xmin, xmax = drawn
        t = (panel_x - PB_X0) / max(PB_X1 - PB_X0, 1)
        return xmin + t * (xmax - xmin)

    def on_mouse(event, x, y, flags, param):
        nonlocal samples_version
        s = disp_state["scale"]
        if s <= 0:
            return

        # ── Events on the panel: wheel = zoom, left-drag = pan ──
        if x >= disp_state["img_w"]:
            panel_x = x - disp_state["img_w"]
            drawn = plot_view["drawn"]
            if drawn is None:
                return
            xmin, xmax = plot_view["range"] or drawn
            if event == cv2.EVENT_MOUSEWHEEL:
                # Zoom around the cursor's data position.
                cx = panel_to_data(panel_x)
                if cx is None:
                    return
                factor = 0.8 if cv2.getMouseWheelDelta(flags) > 0 else 1.25
                plot_view["range"] = (cx - (cx - xmin) * factor,
                                      cx + (xmax - cx) * factor)
            elif event == cv2.EVENT_LBUTTONDOWN:
                plot_view["drag"] = (panel_x, (xmin, xmax))
            elif event == cv2.EVENT_MOUSEMOVE and plot_view["drag"] is not None:
                px0, (vx0, vx1) = plot_view["drag"]
                dx = (panel_x - px0) / max(PB_X1 - PB_X0, 1) * (vx1 - vx0)
                plot_view["range"] = (vx0 - dx, vx1 - dx)
            elif event == cv2.EVENT_LBUTTONUP:
                plot_view["drag"] = None
            return

        # ── Events on the live image: patch selection ──
        if event == cv2.EVENT_LBUTTONUP:
            plot_view["drag"] = None    # drag that wandered off the panel
        if y >= disp_state["img_h"]:
            return                      # status bar
        fx, fy = int(x / s), int(y / s)
        if event == cv2.EVENT_MOUSEMOVE:
            selector.move(fx, fy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            rect = selector.click(fx, fy)
            if rect is not None:
                amp.reset()
                samples.clear()
                samples_version += 1
                x0, y0, x1, y1 = rect
                print(f"Patch selected: x[{x0}:{x1}] y[{y0}:{y1}] "
                      f"({x1 - x0}x{y1 - y0} px)")
                # Record it (scaled to full-res sensor coords) so
                # oct_crop_scan.py crops to exactly this region by default.
                from patch_store import save_patch
                save_patch(x0, y0, x1 - x0, y1 - y0, binning=live_binning)

    cv2.setMouseCallback(window_name, on_mouse)

    # Stream continuously when the camera supports it (avoids per-frame
    # start/stop overhead). Otherwise fall back to single-shot capture().
    streaming = hasattr(cam, "start_streaming")
    if streaming:
        cam.start_streaming(buffer_count=5)

    def grab():
        if streaming:
            return cam.latest_frame()
        return cam.capture()

    print("Starting live view. Click two points to select a patch | c clear | r reset plot")
    print("+/- exposure | w/s (or arrows) move stage, e/d = fine, [ ] step | q quit.")
    print(f"Initial exposure: {exposure_us} µs (non-binned equivalent)  |  "
          f"coarse step: {coarse_mm:g} mm, fine: {coarse_mm / FINE_RATIO:g} mm  |  "
          f"positional window: +/-{window:g} {'mm' if stage is not None else 'frames'}")
    if stage is not None:
        print(f"f = fine scan: +/-{fine_range:g} mm around the current position "
              f"in {fine_step * 1000:.0f} µm steps, saves a contrast plot and "
              "moves to the peak (coarse-align first).")
        # The DC estimate averages patches within +/-window of the current
        # position. Shrink that toward the step size and the DC starts tracking
        # the fringe itself, flattening the very peak the scan is measuring.
        if window < 5 * fine_step:
            print(f"Warning: --window-mm {window:g} is only "
                  f"{window / fine_step:.1f} fine steps wide; the DC estimate "
                  f"will track the fringe and flatten the fine-scan peak. "
                  f"Use >= {5 * fine_step:g}.")

    # FPS measured over a window. When streaming, count real frames delivered by
    # the camera (the loop itself runs faster than frames arrive).
    fps = 0.0
    fps_t0 = time.time()
    fps_count0 = cam.frame_count if streaming else 0

    # Stage position is polled at ~10 Hz (not every frame) to limit serial traffic.
    pos_mm = None
    moving = False
    last_pos_t = 0.0
    last_coord = None        # x-coordinate of the newest sample (for highlight)
    fallback_idx = 0         # x-axis when no stage is connected
    live_val = None

    try:
        while True:
            loop_start = time.time()

            # Grab the latest frame; may be None briefly while streaming spins up
            frame = grab()
            if frame is None:
                cv2.waitKey(1)
                continue

            # Convert to a gamma-corrected 8-bit image so the scene is perceptible
            frame_display = to_display_8bit(frame, gamma)

            # Poll stage position at ~10 Hz
            now = time.time()
            if stage is not None and now - last_pos_t > 0.1:
                try:
                    pos_mm = stage.position
                    moving = stage.is_moving
                except Exception:
                    pass
                last_pos_t = now

            # ── Patch amplitude on the RAW frame (full bit depth, no gamma) ──
            if selector.rect is not None:
                box = clip_rect(selector.rect, frame.shape)
                if box is not None:
                    x0, y0, x1, y1 = box
                    # x-coordinate: stage position if available, else frame index.
                    # While the stage is in motion the 10 Hz cached position is
                    # up to 100 ms stale (directionally biased against the
                    # sweep), so re-poll it for the frame being attributed.
                    if stage is not None and moving:
                        try:
                            pos_mm = stage.position
                            moving = stage.is_moving
                            last_pos_t = now
                        except Exception:
                            pass
                    if stage is not None and pos_mm is not None:
                        coord = pos_mm
                    else:
                        coord = fallback_idx
                        fallback_idx += 1
                    live_val = amp.update(frame[y0:y1, x0:x1], coord)
                    if live_val is not None:
                        # One value per position: upsert on the rounded key
                        # (new keys only admitted below the point cap).
                        key = round(coord, 4)
                        if key in samples or len(samples) < MAX_POINTS:
                            samples[key] = (coord, live_val)
                            samples_version += 1
                        last_coord = coord

            # ── Fine scan state machine: step, dwell, record, advance ──
            if scan is not None and stage is not None:
                tgt = scan["targets"][scan["i"]]
                arrived = (pos_mm is not None and not moving
                           and abs(pos_mm - tgt) < scan["tol"])
                scan["stall"] = 0 if arrived else scan["stall"] + 1
                if arrived:
                    if not scan["cleared"]:
                        # Reset the plot right as the sweep proper begins, so
                        # transit samples from the drive-to-start don't mix in.
                        samples.clear()
                        samples_version += 1
                        amp.reset()
                        last_coord = None
                        live_val = None
                        plot_view["range"] = None
                        scan["cleared"] = True
                    scan["dwell"] += 1
                    if live_val is not None:
                        scan["acc"].append(live_val)
                # Advance once the dwell is done, or give up on a target the
                # stage never settles inside of (tol is only ~10 µm, so a step
                # that lands just outside it would otherwise hang the scan).
                stalled = scan["stall"] > FINE_STALL_FRAMES
                if (arrived and scan["dwell"] >= FINE_DWELL_FRAMES) or stalled:
                    if stalled:
                        print(f"Fine scan: stage never settled at {tgt:.4f} mm; "
                              "skipping that step.")
                    # Record the dwell-averaged contrast at the position actually
                    # measured (not the commanded target).
                    if scan["acc"]:
                        scan["values"].append((pos_mm, float(np.mean(scan["acc"]))))
                    scan["i"] += 1
                    scan["dwell"] = scan["stall"] = 0
                    scan["acc"] = []
                    try:
                        if scan["i"] < len(scan["targets"]):
                            stage.move_to(scan["targets"][scan["i"]], wait=False)
                        else:
                            notice = finish_scan(scan, now)
                            plot_view["range"] = None
                            scan = None
                    except Exception as e:
                        print(f"Fine scan move failed: {e}")
                        scan = None

            # ── Magnified-patch crop — MUST happen before the resize below ──
            # frame_display is overwritten with the screen-fitted downscale; a
            # crop taken from that and magnified back up is interpolated mush,
            # with none of the fringes or speckle the inset exists to show. Uses
            # preview_rect so the inset also tracks the box being dragged out.
            inset = inset_px = None
            box = clip_rect(selector.preview_rect(), frame.shape)
            if box is not None:
                x0, y0, x1, y1 = box
                inset_px = (x1 - x0, y1 - y0)      # true size, before any decimation
                inset = (stretch_patch(frame[y0:y1, x0:x1], gamma) if inset_auto
                         else frame_display[y0:y1, x0:x1])

            # Scale down to fit the screen while preserving aspect ratio.
            # Resize the single-channel image, THEN expand to BGR — at full
            # sensor resolution this is 3x less data through the resize.
            h, w = frame_display.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                frame_display = cv2.resize(
                    frame_display,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)
            disp_state["scale"] = scale
            disp_state["img_w"] = frame_rgb.shape[1]
            disp_state["img_h"] = frame_rgb.shape[0]

            # Patch outline: finalized = series blue, in-progress = live aqua.
            rect = selector.preview_rect()
            if rect is not None:
                x0, y0, x1, y1 = (int(v * scale) for v in rect)
                color = SERIES if selector.rect is not None else LIVE
                cv2.rectangle(frame_rgb, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
            elif selector.anchor is not None:
                ax, ay = (int(v * scale) for v in selector.anchor)
                cv2.drawMarker(frame_rgb, (ax, ay), LIVE, cv2.MARKER_CROSS, 12, 2)

            # Measure actual frame rate over a 0.5 s window
            if streaming:
                if now - fps_t0 >= 0.5:
                    fps = (cam.frame_count - fps_count0) / (now - fps_t0)
                    fps_t0, fps_count0 = now, cam.frame_count
            else:
                loop_time = now - loop_start
                fps = 1.0 / loop_time if loop_time > 0 else 0.0

            # Compose: image | panel, status bar below, then one text flush.
            # Text coordinates are local to each surface, so set the queue
            # origin to that surface's position on the composed canvas.
            txt.origin = (frame_rgb.shape[1], 0)
            panel, drawn = render_panel(txt, list(samples.values()),
                                        frame_rgb.shape[0], live_val,
                                        current_pos=last_coord,
                                        view=plot_view["range"],
                                        has_stage=stage is not None,
                                        cache=panel_cache,
                                        version=samples_version,
                                        inset=inset, inset_auto=inset_auto,
                                        inset_px=inset_px)
            plot_view["drawn"] = drawn
            canvas = np.hstack([frame_rgb, panel])
            # Status-bar banner: fine-scan progress, or a transient notice.
            if notice is not None and now >= notice[1]:
                notice = None
            if entry is not None:
                banner = (f"move to:  {entry}_   mm"
                          "    (Enter to go · Esc to cancel · +/- for a "
                          "relative move)", LIVE)
            elif scan is not None:
                banner = (f"FINE SCAN  {scan['i'] + 1}/{len(scan['targets'])}"
                          f"   target {scan['targets'][scan['i']]:.3f} mm"
                          "   (f cancels)", ACCENT)
            elif notice is not None:
                banner = (notice[0], ACCENT)
            else:
                banner = None

            txt.origin = (0, frame_rgb.shape[0])
            bar = render_status(txt, canvas.shape[1], exposure_us, gamma, fps,
                                live_binning, stage is not None, pos_mm,
                                moving, coarse_mm, banner=banner)
            canvas = np.vstack([canvas, bar])
            txt.origin = (0, 0)
            canvas = txt.flush(canvas)

            # Display
            cv2.imshow(window_name, canvas)
            if not window_sized:
                cv2.resizeWindow(window_name, canvas.shape[1], canvas.shape[0])
                window_sized = True

            # Handle keyboard input (waitKeyEx preserves arrow-key codes)
            key = cv2.waitKeyEx(1)
            if key != -1:
                k = key & 0xFF
                # ── Typed position entry ('m'): consumes EVERY key while active.
                # This must come first: the normal bindings below include s, d,
                # e, w, -, [, ] and q, so without it typing "3.75" would jog the
                # stage and typing a stray letter could quit the program.
                if entry is not None:
                    if k in (13, 10):                       # Enter — go
                        try:
                            target, rel = resolve_target(entry, pos_mm)
                            stage.move_to(target, wait=False)
                            notice = (f"moving to {target:.4f} mm", now + 4)
                            print(f"Moving to {target:.4f} mm"
                                  + (f" ({entry} from {pos_mm:.4f})" if rel else ""))
                        except ValueError as e:
                            notice = (f"move: {e}", now + 5)
                            print(f"Move refused: {e}")
                        except Exception as e:
                            notice = (f"move failed: {e}", now + 5)
                            print(f"Move failed: {e}")
                        entry = None
                    elif k == 27:                           # Esc — cancel
                        entry = None
                    elif k in (8, 127):                     # Backspace
                        entry = entry[:-1]
                    elif chr(k) in "0123456789." or (chr(k) in "+-" and not entry):
                        entry += chr(k)                     # leading +/- = relative
                    # anything else is ignored rather than acted on
                # Stage: w/s (or arrows) = coarse; e/d = fine (coarse / FINE_RATIO).
                elif key in UP_KEYS or k == ord('w'):
                    move_stage(+1)
                elif key in DOWN_KEYS or k == ord('s'):
                    move_stage(-1)
                elif k == ord('e'):
                    move_stage(+1, fine=True)
                elif k == ord('d'):
                    move_stage(-1, fine=True)
                elif k == ord('f'):
                    if scan is not None:
                        scan = None
                        notice = ("fine scan cancelled", now + 3)
                        print("Fine scan cancelled.")
                    elif stage is None:
                        notice = ("fine scan needs a stage", now + 4)
                    elif selector.rect is None:
                        notice = ("fine scan: select a patch first (two clicks)",
                                  now + 4)
                    elif pos_mm is None or moving:
                        # The window is anchored to the current position, and a
                        # mid-move poll is up to 100 ms stale.
                        notice = ("fine scan: wait for the stage to stop", now + 4)
                    else:
                        # Clamp into the travel range: a scan near an end becomes
                        # asymmetric rather than commanding an invalid target.
                        lo = max(0.0, pos_mm - fine_range)
                        hi = min(STAGE_MAX_MM, pos_mm + fine_range)
                        n_steps = int(round((hi - lo) / fine_step)) + 1
                        if n_steps < FINE_MIN_STEPS:
                            notice = ("fine scan: too close to a travel limit",
                                      now + 5)
                            print(f"Fine scan refused: only {n_steps} steps fit "
                                  f"between {lo:.3f} and {hi:.3f} mm.")
                        else:
                            targets = [min(lo + i * fine_step, hi)
                                       for i in range(n_steps)]
                            est_s = n_steps * (FINE_DWELL_FRAMES
                                               / max(fps, 1.0) + 0.3)
                            scan = start_scan(targets, fine_step)
                            if scan is not None:
                                print(f"Fine scan: {lo:.3f} → {hi:.3f} mm in "
                                      f"{n_steps} steps of "
                                      f"{fine_step * 1000:.0f} µm "
                                      f"(est ~{est_s:.0f}s). Press f to cancel.")
                elif k == ord('i'):
                    inset_auto = not inset_auto
                    print("Patch inset: auto-contrast (stretched — exaggerates "
                          "contrast and hides saturation)" if inset_auto else
                          "Patch inset: true levels (matches the live view)")
                elif k == ord('m'):
                    if stage is None:
                        notice = ("no stage to move", now + 4)
                    elif scan is not None:
                        notice = ("finish or cancel the fine scan first", now + 4)
                    elif pos_mm is None:
                        notice = ("stage position unknown", now + 4)
                    else:
                        entry = ""      # every key now goes to the entry branch
                elif k == ord('c'):
                    selector.clear()
                    amp.reset()
                    samples.clear()
                    samples_version += 1
                    last_coord = None
                    fallback_idx = 0
                    live_val = None
                    plot_view["range"] = None
                    scan = None
                    print("Patch selection cleared.")
                elif k == ord('r'):
                    samples.clear()
                    samples_version += 1
                    last_coord = None
                    fallback_idx = 0
                    plot_view["range"] = None
                    print("Plot data reset.")
                elif k == ord('a'):
                    plot_view["range"] = None
                    print("Plot autoscaled.")
                elif k == ord('q'):
                    print("Exiting...")
                    break
                elif k == ord('+') or k == ord('='):
                    exposure_us = min(exposure_us + exposure_step, 1000000)  # Cap at 1s
                    apply_exposure(exposure_us)
                    print(f"Exposure: {exposure_us} µs")
                elif k == ord('-') or k == ord('_'):
                    exposure_us = max(exposure_us - exposure_step, 100)  # Floor at 100 µs
                    apply_exposure(exposure_us)
                    print(f"Exposure: {exposure_us} µs")
                elif k == ord('['):
                    coarse_mm = max(coarse_mm / 2, 0.001)
                    print(f"Coarse step: {coarse_mm:g} mm (fine {coarse_mm / FINE_RATIO:g} mm)")
                elif k == ord(']'):
                    coarse_mm = min(coarse_mm * 2, 50.0)
                    print(f"Coarse step: {coarse_mm:g} mm (fine {coarse_mm / FINE_RATIO:g} mm)")

            # Frame rate control (only sleep if capture was faster than target)
            total_time = time.time() - loop_start
            sleep_time = max(0, frame_time_target - total_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # Stop streaming before changing binning (binning needs a stopped stream).
        if streaming:
            try:
                cam.stop_streaming()
            except Exception:
                pass
        # Restore full resolution so later full-res captures aren't left binned.
        try:
            cam.set_binning(1)
        except Exception:
            pass
        cam.release()
        if stage is not None:
            try:
                stage.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
