#!/usr/bin/env python3
"""oct_contrast_video.py — side-by-side movie of an OCT scan and its contrast.

Left: the raw captured frames (globally normalized). Right: the
compute_mean_diff interference amplitude |I - DC| for the same frames — the
quantity whose per-pixel argmax becomes the depth map. A header labels the two
panels; a footer tracks the frame index and stage position (read from the
_meta.json sidecar when present).

Usage:
    python oct_contrast_video.py -s setupA_test/stack_20260709_024053.npy
    python oct_contrast_video.py -s stack.npy -n 2 --fps 30 -o out.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from oct import compute_mean_diff

# ── Defaults (overridable via command-line args) ───────────────────────────────
DOWNSAMPLE = 2       # spatially average NxN patches before processing
FPS = 30.0
PATCH_SIZE = 5       # compute_mean_diff spatial patch (matches oct_process.py)
TEMPORAL_WINDOW = 20  # compute_mean_diff DC window (matches oct_process.py)

HEADER_H, FOOTER_H, GAP = 46, 40, 4
BG, TEXT_1, TEXT_2 = 25, 245, 170  # grayscale levels for chrome and text


def load_font(size):
    """Prefer the same geometric faces the visualizer uses; fall back cleanly."""
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/Avenir Next.ttc",
                 "/System/Library/Fonts/Avenir.ttc",
                 "C:/Windows/Fonts/segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def downsample(frames, n):
    """Block-average NxN over the spatial axes, one frame at a time (low RAM)."""
    if n <= 1:
        return frames.astype(np.float32)
    h, w, N = frames.shape
    h2, w2 = (h // n) * n, (w // n) * n
    out = np.empty((h2 // n, w2 // n, N), np.float32)
    for i in range(N):
        f = frames[:h2, :w2, i].astype(np.float32)
        out[:, :, i] = f.reshape(h2 // n, n, w2 // n, n).mean(axis=(1, 3))
    return out


def normalize(stack, lo_pct, hi_pct):
    """Map [lo_pct, hi_pct] percentiles of the whole stack to 0..255 uint8."""
    lo, hi = np.percentile(stack, [lo_pct, hi_pct])
    scaled = np.clip((stack - lo) / max(hi - lo, 1e-9), 0, 1)
    return (scaled * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(
        description="Render an OCT stack and its compute_mean_diff contrast side by side.")
    p.add_argument("-s", "--stack", type=Path, required=True, help="path to the stack .npy")
    p.add_argument("-n", "--downsample", type=int, default=DOWNSAMPLE,
                   help=f"spatially average NxN patches (default {DOWNSAMPLE})")
    p.add_argument("--fps", type=float, default=FPS, help=f"output frame rate (default {FPS:g})")
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    p.add_argument("--temporal-window", type=int, default=TEMPORAL_WINDOW)
    p.add_argument("--norm", choices=["per-pixel", "global"], default="per-pixel",
                   help="contrast normalization: 'per-pixel' divides each pixel's "
                        "time series by its own max, so every pixel peaks at "
                        "full brightness at its in-focus depth (default); "
                        "'global' uses one scale for the whole stack")
    p.add_argument("--smooth-frames", type=int, default=3,
                   help="temporal box smoothing of the contrast envelope, in frames "
                        "(default 3; 1 disables). Keep it well under the coherence "
                        "FWHM so the envelope isn't materially widened.")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output .mp4 (default <stack>_contrast.mp4)")
    args = p.parse_args()

    out_path = args.out or args.stack.with_name(args.stack.stem + "_contrast.mp4")

    print(f"Loading: {args.stack}")
    frames = np.load(args.stack, mmap_mode="r")
    print(f"  shape={frames.shape}  dtype={frames.dtype}")

    positions = None
    meta_path = args.stack.with_name(args.stack.stem + "_meta.json")
    if meta_path.exists():
        positions = json.loads(meta_path.read_text()).get("positions_mm")

    frames = downsample(frames, args.downsample)
    h, w, N = frames.shape
    print(f"  processing at {h}x{w}x{N}")

    print("Running compute_mean_diff...")
    _, md_image, _ = compute_mean_diff(frames, patch_size=args.patch_size,
                                       avg_type="local",
                                       temporal_window=args.temporal_window)

    raw8 = normalize(frames, 0.1, 99.9)

    if args.smooth_frames > 1:
        # Light temporal smoothing of the envelope to suppress frame-to-frame
        # speckle flicker (the envelope itself varies on the coherence scale,
        # which is much slower than this window).
        from scipy.ndimage import uniform_filter1d
        md_image = uniform_filter1d(md_image, size=args.smooth_frames,
                                    axis=2, mode="nearest")

    if args.norm == "per-pixel":
        # Subtract each pixel's noise floor (its temporal median), then scale
        # by its max: signal pixels still peak at full brightness exactly at
        # their in-focus depth, while empty pixels stay dark instead of having
        # their noise blown up to full range.
        floor = np.median(md_image, axis=2, keepdims=True)
        peak = md_image.max(axis=2, keepdims=True)
        amp = (md_image - floor) / np.maximum(peak - floor, 1e-9)
        amp8 = (np.clip(amp, 0, 1) * 255).astype(np.uint8)
    else:
        amp8 = normalize(md_image, 0.0, 99.8)  # amplitude is spiky — clip the top tail
    del frames, md_image

    # ── Compose: header | raw + contrast | footer ──────────────────────────────
    from PIL import Image, ImageDraw
    W, H = 2 * w + GAP, HEADER_H + h + FOOTER_H
    font_l, font_f = load_font(24), load_font(20)

    header = Image.new("L", (W, HEADER_H), BG)
    d = ImageDraw.Draw(header)
    for label, cx in (("Raw scan frames", w // 2),
                      ("computeMeanDiff contrast  |I − DC|", w + GAP + w // 2)):
        tw = d.textlength(label, font=font_l)
        d.text((cx - tw / 2, (HEADER_H - 30) / 2), label, fill=TEXT_1, font=font_l)
    header = np.asarray(header)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H), isColor=False)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")

    for i in range(N):
        canvas = np.full((H, W), BG, np.uint8)
        canvas[:HEADER_H] = header
        canvas[HEADER_H:HEADER_H + h, :w] = raw8[:, :, i]
        canvas[HEADER_H:HEADER_H + h, w + GAP:] = amp8[:, :, i]

        footer = Image.new("L", (W, FOOTER_H), BG)
        d = ImageDraw.Draw(footer)
        label = f"frame {i + 1}/{N}"
        if positions:
            label += f"    stage {positions[i]:.4f} mm"
        tw = d.textlength(label, font=font_f)
        d.text(((W - tw) / 2, (FOOTER_H - 26) / 2), label, fill=TEXT_2, font=font_f)
        canvas[HEADER_H + h:] = np.asarray(footer)

        writer.write(canvas)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{N}]")

    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
