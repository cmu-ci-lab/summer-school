#!/usr/bin/env python3
"""
oct_scan.py — capture an OCT Z-stack.

Homes the stage (unless --no-home), sweeps it in fixed steps across a range
centred on --center (take the centre from the visualizer's peak readout),
captures one frame per position, and saves the stack as a timestamped .npy
plus a _meta.json sidecar describing the scan geometry (used by
depth_to_pointcloud.py and friends). The frame count is range/step + 1.

If --exposure is not given, the scan keeps whatever exposure the sensor is
currently set to — so an exposure dialed in with the visualizer carries over.

Usage:
    python oct_scan.py --center 1.70 --range-mm 0.4 --step-mm 0.001   # 401 frames
    python oct_scan.py --center 1.70 --range-mm 0.4 --step-mm 0.002 \
        --exposure 8000 --save-dir coin_scan
    python oct_scan.py -n 2                 # 2x2 spatial downsample at capture
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
    p.add_argument("--center", type=float, default=1.70,
                   help="centre of the scan in mm — use the peak position "
                        "from the visualizer (default 1.70)")
    p.add_argument("--range-mm", type=float, default=0.4,
                   help="total scan range in mm, centred on --center "
                        "(default 0.4)")
    p.add_argument("--step-mm", type=float, default=0.001,
                   help="stage step per frame in mm (default 0.001); the "
                        "frame count is range/step + 1")
    p.add_argument("-n", "--downsample", type=int, default=1,
                   help="spatially average NxN pixel patches per captured "
                        "frame before saving (default 1 = full resolution); "
                        "shrinks the stack by N^2 and is recorded in the "
                        "metadata so the lateral scale stays correct")
    p.add_argument("--exposure", type=float, default=None,
                   help="exposure in microseconds (default: the value from "
                        "your last visualizer session — last_exposure.json — "
                        "else whatever the sensor is currently set to)")
    p.add_argument("--gain", type=float, default=0.0,
                   help="camera gain (default 0)")
    p.add_argument("--save-dir", default="oct_scans",
                   help="output directory for the stack + metadata (default oct_scans)")
    p.add_argument("--camera", choices=["auto", "avt", "ids"], default="auto",
                   help="camera vendor (default auto-detect)")
    p.add_argument("--no-home", action="store_true",
                   help="skip homing the stage first")
    args = p.parse_args()

    # Centre + range -> start position and frame count.
    n_frames = int(round(args.range_mm / args.step_mm)) + 1
    if n_frames < 2:
        raise SystemExit(f"Scan range {args.range_mm:g} mm at {args.step_mm:g} mm "
                         "steps gives fewer than 2 frames.")
    start = args.center - args.range_mm / 2
    if start < 0:
        raise SystemExit(f"Scan would start at {start:g} mm (< 0): centre "
                         f"{args.center:g} mm with range {args.range_mm:g} mm.")
    end = start + (n_frames - 1) * args.step_mm
    ds = max(args.downsample, 1)

    # Exposure: CLI value > last visualizer session (last_exposure.json) >
    # whatever the sensor currently reports. The file exists because the IDS
    # driver resets exposure to ~1/framerate on re-init, so the sensor itself
    # cannot carry a visualizer-dialed value between programs.
    exposure = args.exposure
    if exposure is None:
        from exposure_store import load_exposure
        stored = load_exposure()
        if stored is not None:
            exposure, age = stored
            print(f"Using exposure from your last visualizer session: "
                  f"{exposure:g} µs  (saved {age})")
    exp_s = f"{exposure:g} us" if exposure is not None else "current sensor value"
    print(f"Scan: {n_frames} frames x {args.step_mm:g} mm  "
          f"({start:g} -> {end:g} mm, centre {args.center:g})   exposure {exp_s}"
          + (f"   downsample {ds}x{ds}" if ds > 1 else ""))

    # Hardware imports deferred so --help works without the drivers installed.
    from stage import ThorlabsStage
    from camera import Camera

    stage = ThorlabsStage(units="mm")
    stage.connect()
    cam = Camera(exposure_us=exposure, gain_db=args.gain,
                 save_dir=args.save_dir,
                 prefer=None if args.camera == "auto" else args.camera)
    cam.connect()

    try:
        if not args.no_home:
            stage.home()

        positions = []
        frames = []
        for i in range(n_frames):
            stage.move_to(start + i * args.step_mm)
            frame = cam.capture()
            if ds > 1:
                # Spatial NxN mean, rounded back to the sensor dtype so a
                # long scan doesn't balloon into a float64 stack.
                frame = np.rint(downsample_spatial(frame, ds)).astype(frame.dtype)
            frames.append(frame)
            positions.append(stage.position)
            print(f"  [{i + 1}/{n_frames}]  {stage.position:.4f} mm")

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
            "start_position_mm": start,
            "center_position_mm": args.center,
            "range_mm": args.range_mm,
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
