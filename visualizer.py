#!/usr/bin/env python3
"""
visualizer.py — live camera + stage viewer with a coherence-envelope panel.

Successor to live_view_coherence.py with a redesigned interface: dark theme,
Avenir (or the closest geometric face installed) for all text via Pillow, stat
tiles for the live readouts, and a cleaner plot.

Select a rectangular patch with two clicks; its interference amplitude (the
compute_mean_diff measure: mean |frame - local mean| over the patch) is plotted
against stage position. The local (DC) mean is positional — patches captured
within +/- --window-mm of the current stage position — so irregular stage
motion is handled naturally, and the plot keeps exactly one value per position.
Once the envelope crosses half-max on both sides of the peak, the coherence
length (FWHM after floor subtraction) and the peak location are shown.

Controls:
  click x2   : Select patch (first click anchors a corner, second finalizes)
  wheel      : (over the plot) zoom the position axis around the cursor
  left-drag  : (over the plot) pan the position axis
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
"""

import cv2
import argparse
import numpy as np
from collections import deque
import time

# Arrow-key codes vary by OS/GUI backend; match against all known up/down values
# (macOS Qt, GTK, Windows). w/s are provided as backend-independent alternates.
UP_KEYS   = {63232, 65362, 2490368, 82}
DOWN_KEYS = {63233, 65364, 2621440, 84}

# e/d move a finer step, FINE_RATIO times smaller than the w/s coarse step.
FINE_RATIO = 10

MAX_POINTS = 5000         # cap on stored (position, amplitude) samples

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
        """Queue a string; `anchor` is a PIL text anchor (la, ra, ma, ...)."""
        xy = (xy[0] + self.origin[0], xy[1] + self.origin[1])
        self.queue.append((xy, s, size, color, weight, anchor))

    def width(self, s, size, weight="regular"):
        """Rendered width of `s` in pixels (estimate under the cv2 fallback)."""
        if not self.ok:
            return len(s) * size * 0.55
        return self._font(size, weight).getlength(s)

    def flush(self, canvas):
        """Draw all queued strings onto the BGR canvas; returns the canvas."""
        if not self.queue:
            return canvas
        if not self.ok:                      # Hershey fallback
            for (x, y), s, size, color, weight, anchor in self.queue:
                if anchor.startswith("r"):
                    x -= int(len(s) * size * 0.55)
                elif anchor.startswith("m"):
                    x -= int(len(s) * size * 0.28)
                cv2.putText(canvas, s, (int(x), int(y + size)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 26.0, color, 1,
                            cv2.LINE_AA)
            self.queue.clear()
            return canvas
        img = self.Image.fromarray(canvas[:, :, ::-1])   # BGR -> RGB
        draw = self.ImageDraw.Draw(img)
        for xy, s, size, color, weight, anchor in self.queue:
            draw.text(xy, s, font=self._font(size, weight),
                      fill=color[::-1], anchor=anchor)     # BGR -> RGB
        self.queue.clear()
        return np.asarray(img)[:, :, ::-1].copy()          # RGB -> BGR


def to_display_8bit(frame, gamma):
    """Convert a raw frame to a gamma-corrected 8-bit image for human viewing.

    The data is normalized to [0, 1] by its bit depth, then encoded with the
    standard display gamma (out = in**(1/gamma); gamma > 1 brightens the shadows
    so a dark linear scene becomes visible), then scaled to 8-bit. Gamma is
    applied on the full-bit-depth data, before quantizing, so shadow detail
    isn't lost to an early bit-shift.
    """
    if frame.dtype == np.uint8:
        norm = frame.astype(np.float32) / 255.0
    elif frame.dtype == np.uint16:
        # 12-bit sensors store values <= 4095 in a 16-bit container.
        maxv = 4095.0 if (frame.size and int(frame.max()) <= 4095) else 65535.0
        norm = frame.astype(np.float32) / maxv
    else:
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
        stage.home()
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

    def reset(self):
        self.buf.clear()

    def update(self, patch, pos):
        """Add a (pos, patch) sample; return the amplitude at `pos`, or None.

        `pos` may be any monotone-ish coordinate — stage mm normally, or a
        frame index (with window_mm reinterpreted in frames) when no stage.
        """
        patch = patch.astype(np.float64)
        # A patch-size change (new selection) invalidates the buffer.
        if self.buf and self.buf[0][1].shape != patch.shape:
            self.buf.clear()
        self.buf.append((pos, patch))

        # DC estimate: mean of the patches whose position falls in the window.
        near = [p for (q, p) in self.buf if abs(q - pos) <= self.window_mm]
        if len(near) < 2:
            return None
        avg = np.mean(near, axis=0)
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


def render_panel(txt, points, height, live_val=None, current_pos=None,
                 view=None, has_stage=True):
    """Render the right-hand panel: header, stat tiles, and the plot.

    `points` is a sequence of (position, amplitude), one value per position.
    `view` is an optional (xmin, xmax) zoom range; None = auto-fit all data.
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
    fwhm = None
    px = py = None
    if len(points) >= 2:
        pts = np.asarray(points, dtype=np.float64)
        order = np.argsort(pts[:, 0])
        px, py = pts[order, 0], pts[order, 1]
        fwhm = compute_fwhm(px, py)

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

    # ── Plot box ──
    top = HEADER_H + TILES_H
    bot = height - 34
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

    # Curve: 2px antialiased polyline through the visible points.
    pts_px = [to_px(px[i], py[i]) for i in range(len(px))]
    seg = [p for p in pts_px if PB_X0 <= p[0] <= PB_X1]
    if len(seg) >= 2:
        cv2.polylines(panel, [np.asarray(seg, np.int32)], False, SERIES, 2,
                      cv2.LINE_AA)
    for p in seg:
        cv2.circle(panel, p, 2, SERIES, -1, cv2.LINE_AA)

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
                  pos_mm, moving, coarse_mm):
    """Full-width bottom status bar: camera stats left, stage centre, keys right."""
    bar = np.full((STATUS_H, width, 3), SURFACE_2, np.uint8)
    cv2.line(bar, (0, 0), (width, 0), GRID, 1)
    cy = STATUS_H // 2

    # "gamma" spelled out: Ageo has no Greek glyphs, so a γ would drop out.
    cam_s = f"{int(exposure_us)} µs    gamma {gamma:g}    {fps:0.1f} fps    bin {binning}×"
    txt.put((20, cy), cam_s, 16, TEXT_1, "demi", anchor="lm")

    if stage_ok:
        pos_s = f"{pos_mm:.4f} mm" if pos_mm is not None else "—"
        stage_s = f"stage {pos_s}    step {coarse_mm:g} / {coarse_mm / FINE_RATIO:g} mm"
        color = LIVE if moving else TEXT_1
        if moving:
            stage_s += "    MOVING"
    else:
        stage_s, color = "stage not found — camera only", TEXT_MUTED
    txt.put((width // 2, cy), stage_s, 16, color, "demi", anchor="mm")

    txt.put((width - 20, cy), "w/s move   e/d fine   [ ] step   c clear   a fit   q quit",
            11, TEXT_MUTED, anchor="rm")
    return bar


def main():
    parser = argparse.ArgumentParser(
        description="Live camera viewer with stage control and coherence panel.")
    parser.add_argument("--step", type=float, default=1.0,
                        help="coarse stage step in mm (w/s or arrows); e/d "
                             f"moves 1/{FINE_RATIO} of this (default 1.0)")
    parser.add_argument("--binning", type=int, default=4,
                        help="on-sensor binning factor for the preview (default 4)")
    parser.add_argument("--exposure", type=float, default=10000,
                        help="initial exposure in microseconds, non-binned-equivalent (default 10000)")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="display gamma; >1 brightens dark scenes (default 2.2, 1.0 = linear)")
    parser.add_argument("--window-mm", type=float, default=0.2,
                        help="half-width in mm of the positional window used for "
                             "the DC (local mean) estimate; patches captured "
                             "within +/- this distance of the current position "
                             "are averaged (default 0.2). With no stage it is "
                             "interpreted in frames.")
    args = parser.parse_args()
    gamma = args.gamma

    txt = Text()

    print("Initializing camera...")
    from camera import Camera
    cam = Camera(exposure_us=args.exposure, gain_db=0.0, save_dir="live_view_captures")
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
        """Set the camera exposure compensated for binning brightness gain."""
        try:
            cam.set_exposure(max(shown_us / (live_binning ** 2), 1))
        except Exception as e:
            print(f"Could not set exposure: {e}")

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
    # plot_view["range"]: (xmin, xmax) zoom of the plot, None = auto-fit.
    # plot_view["drawn"]: the range actually rendered last frame (needed to map
    # panel pixels back to data coordinates for wheel-zoom / drag-pan).
    plot_view = {"range": None, "drawn": None, "drag": None}
    disp_state = {"scale": 1.0, "img_w": 0, "img_h": 0}

    def panel_to_data(panel_x):
        """Map an x pixel inside the plot panel to a data (stage) coordinate."""
        drawn = plot_view["drawn"]
        if drawn is None:
            return None
        xmin, xmax = drawn
        t = (panel_x - PB_X0) / max(PB_X1 - PB_X0, 1)
        return xmin + t * (xmax - xmin)

    def on_mouse(event, x, y, flags, param):
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
                x0, y0, x1, y1 = rect
                print(f"Patch selected: x[{x0}:{x1}] y[{y0}:{y1}] "
                      f"({x1 - x0}x{y1 - y0} px)")

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

            # Convert to RGB for display (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_GRAY2BGR)

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
                x0, y0, x1, y1 = selector.rect
                fh, fw = frame.shape[:2]
                x0, x1 = np.clip((x0, x1), 0, fw)
                y0, y1 = np.clip((y0, y1), 0, fh)
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    # x-coordinate: stage position if available, else frame index
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
                        last_coord = coord

            # Scale down to fit the screen while preserving aspect ratio
            h, w = frame_rgb.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
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
                                        has_stage=stage is not None)
            plot_view["drawn"] = drawn
            canvas = np.hstack([frame_rgb, panel])
            txt.origin = (0, frame_rgb.shape[0])
            bar = render_status(txt, canvas.shape[1], exposure_us, gamma, fps,
                                live_binning, stage is not None, pos_mm,
                                moving, coarse_mm)
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
                # Stage: w/s (or arrows) = coarse; e/d = fine (coarse / FINE_RATIO).
                if key in UP_KEYS or k == ord('w'):
                    move_stage(+1)
                elif key in DOWN_KEYS or k == ord('s'):
                    move_stage(-1)
                elif k == ord('e'):
                    move_stage(+1, fine=True)
                elif k == ord('d'):
                    move_stage(-1, fine=True)
                elif k == ord('c'):
                    selector.clear()
                    amp.reset()
                    samples.clear()
                    last_coord = None
                    fallback_idx = 0
                    live_val = None
                    plot_view["range"] = None
                    print("Patch selection cleared.")
                elif k == ord('r'):
                    samples.clear()
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
