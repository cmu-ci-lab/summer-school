#!/usr/bin/env python3
"""read_stack.py — convert a captured .npy frame stack into a movie.

The stack is (H, W, N): N frames along the last axis. Each frame is rescaled to
8-bit and written as a video. By default the rescale uses the whole-stack
min/max so brightness is consistent across the movie; pass --per-frame to
autoscale each frame independently.

Encoding defaults to lossless FFV1 (minimal compression, .mkv container).
Pass --lossy for a smaller, compressed .mp4 (mp4v).

Usage:
    python read_stack.py                              # lossless .mkv, defaults below
    python read_stack.py -s path/to/stack.npy --fps 30
    python read_stack.py --lossy -o out.mp4           # compressed mp4
    python read_stack.py --per-frame                  # autoscale each frame
"""

import argparse
from pathlib import Path
import numpy as np
import cv2

# ── Defaults (overridable via command-line args) ───────────────────────────────
STACK_PATH = Path("coin_captures_ids_3250/stack.npy")
FPS = 20.0
LOSSLESS_CODEC, LOSSLESS_EXT = "FFV1", ".mkv"   # lossless — minimal compression
LOSSY_CODEC, LOSSY_EXT = "mp4v", ".mp4"         # compressed fallback


def to_8bit(arr, lo, hi):
    """Rescale arr from [lo, hi] to an 8-bit image."""
    if arr.dtype == np.uint8:
        return arr
    if hi > lo:
        out = (arr.astype(np.float32) - lo) / (hi - lo) * 255.0
    else:
        out = np.zeros(arr.shape, np.float32)
    return out.clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="Convert an .npy frame stack into a movie.")
    p.add_argument("-s", "--stack", type=Path, default=STACK_PATH,
                   help=f"path to the stack .npy (default {STACK_PATH})")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="output video path (default: <stack> with the codec's extension)")
    p.add_argument("--fps", type=float, default=FPS,
                   help=f"frames per second (default {FPS})")
    p.add_argument("--per-frame", action="store_true",
                   help="rescale each frame independently instead of whole-stack")
    p.add_argument("--lossy", action="store_true",
                   help="use compressed mp4 (mp4v) instead of lossless FFV1")
    p.add_argument("--quality", type=float, default=100.0,
                   help="encoder quality 0-100 (default 100)")
    args = p.parse_args()

    codec, ext = (LOSSY_CODEC, LOSSY_EXT) if args.lossy else (LOSSLESS_CODEC, LOSSLESS_EXT)

    stack = np.load(args.stack)
    print(f"Loaded: {args.stack}  shape={stack.shape}  dtype={stack.dtype}")

    if stack.ndim == 2:            # single image -> one-frame movie
        stack = stack[:, :, None]
    if stack.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or 3D (H, W, N) stack, got shape {stack.shape}."
        )

    h, w, n = stack.shape

    # The codec dictates the container: force the right extension so the
    # encoder/muxer combination is valid (e.g. FFV1 needs .mkv, not .mp4).
    out_path = args.out or args.stack
    if out_path.suffix.lower() != ext:
        if args.out is not None:
            print(f"Note: {codec} needs a {ext} container; using {out_path.with_suffix(ext).name}")
        out_path = out_path.with_suffix(ext)

    lo, hi = float(stack.min()), float(stack.max())
    scale = "per-frame" if args.per_frame else f"global ({lo:g}-{hi:g})"
    kind = "lossy" if args.lossy else "lossless"
    print(f"Frames: {n}  size: {w}x{h}  rescale: {scale}  codec: {codec} ({kind})")

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*codec), args.fps, (w, h), isColor=True
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter ({codec}) for {out_path}")
    if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, args.quality)

    try:
        for k in range(n):
            frame = stack[:, :, k]
            if args.per_frame:
                frame8 = to_8bit(frame, float(frame.min()), float(frame.max()))
            else:
                frame8 = to_8bit(frame, lo, hi)
            writer.write(cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR))
            print(f"\r  wrote frame {k + 1}/{n}", end="", flush=True)
        print()
    finally:
        writer.release()

    print(f"Saved movie: {out_path}  ({n} frames @ {args.fps:g} fps)")


if __name__ == "__main__":
    main()
