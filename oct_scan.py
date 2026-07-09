#!/usr/bin/env python3
"""
oct_scan.py — capture an OCT Z-stack.

Homes the stage (unless --no-home), sweeps it in fixed steps from a start
position, captures one frame per position, and saves the stack as a
timestamped .npy plus a _meta.json sidecar describing the scan geometry
(used by depth_to_pointcloud.py and friends).

Usage:
    python oct_scan.py --start 1.50 --step-mm 0.001 --frames 400
    python oct_scan.py --start 1.50 --step-mm 0.002 --frames 200 \
        --exposure 8000 --save-dir coin_scan
    python oct_scan.py -n 2 --frames 400    # 2x2 spatial downsample at capture
    python oct_scan.py ... --no-home        # stage already homed this session
"""

import argparse
import json
import numpy as np
from pathlib import Path

from oct import downsample_spatial


def main():
    p = argparse.ArgumentParser(
        description="Capture an OCT Z-stack: sweep the stage and save a frame stack.")
    p.add_argument("--start", type=float, default=1.50,
                   help="start position in mm (default 1.50)")
    p.add_argument("--step-mm", type=float, default=0.001,
                   help="stage step per frame in mm (default 0.001)")
    p.add_argument("--frames", type=int, default=400,
                   help="number of frames to capture (default 400)")
    p.add_argument("-n", "--downsample", type=int, default=1,
                   help="spatially average NxN pixel patches per captured "
                        "frame before saving (default 1 = full resolution); "
                        "shrinks the stack by N^2 and is recorded in the "
                        "metadata so the lateral scale stays correct")
    p.add_argument("--exposure", type=float, default=8000,
                   help="exposure in microseconds (default 8000)")
    p.add_argument("--gain", type=float, default=0.0,
                   help="camera gain (default 0)")
    p.add_argument("--save-dir", default="oct_scans",
                   help="output directory for the stack + metadata (default oct_scans)")
    p.add_argument("--camera", choices=["auto", "avt", "ids"], default="auto",
                   help="camera vendor (default auto-detect)")
    p.add_argument("--no-home", action="store_true",
                   help="skip homing the stage first")
    args = p.parse_args()

    end = args.start + (args.frames - 1) * args.step_mm
    ds = max(args.downsample, 1)
    print(f"Scan: {args.frames} frames x {args.step_mm:g} mm  "
          f"({args.start:g} -> {end:g} mm)   exposure {args.exposure:g} us"
          + (f"   downsample {ds}x{ds}" if ds > 1 else ""))

    # Hardware imports deferred so --help works without the drivers installed.
    from stage import ThorlabsStage
    from camera import Camera

    stage = ThorlabsStage(units="mm")
    stage.connect()
    cam = Camera(exposure_us=args.exposure, gain_db=args.gain,
                 save_dir=args.save_dir,
                 prefer=None if args.camera == "auto" else args.camera)
    cam.connect()

    try:
        if not args.no_home:
            stage.home()

        positions = []
        frames = []
        for i in range(args.frames):
            stage.move_to(args.start + i * args.step_mm)
            frame = cam.capture()
            if ds > 1:
                # Spatial NxN mean, rounded back to the sensor dtype so a
                # long scan doesn't balloon into a float64 stack.
                frame = np.rint(downsample_spatial(frame, ds)).astype(frame.dtype)
            frames.append(frame)
            positions.append(stage.position)
            print(f"  [{i + 1}/{args.frames}]  {stage.position:.4f} mm")

        stack = np.stack(frames, axis=-1)
        stack_name = cam.timestamped_filename(prefix="stack", ext="npy")
        stack_path = Path(cam.save_dir) / stack_name
        np.save(stack_path, stack)
        print(f"Saved stack: {stack_path}  shape={stack.shape}  dtype={stack.dtype}")

        # Sidecar metadata so downstream tools (e.g. depth_to_pointcloud.py)
        # know the scan geometry without re-connecting the hardware.
        # pixel_size_um describes the SAVED stack (sensor pitch x downsample);
        # the native sensor pitch is kept alongside for reference.
        sensor_um = getattr(cam, "pixel_size_um", None)
        meta = {
            "step_mm": args.step_mm,
            "start_position_mm": args.start,
            "n_frames": len(positions),
            "positions_mm": positions,
            "exposure_us": cam.exposure_us,
            "downsample": ds,
            "pixel_size_um": sensor_um * ds if sensor_um else None,
            "sensor_pixel_size_um": sensor_um,
        }
        meta_path = stack_path.with_name(stack_path.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved metadata: {meta_path}")
    finally:
        stage.release()
        cam.release()


if __name__ == "__main__":
    main()
