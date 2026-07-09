#!/usr/bin/env python3
"""
depth_to_pointcloud.py — turn a depth-index map (from oct_process.py) into a
metric point cloud (.ply).

Geometry:
  * Lateral (x, y): the system is ~1:1 magnification, so a pixel on the sensor
    is a pixel on the object — the scale is the sensor pixel pitch. The pitch
    is taken from (in priority order): --pixel-um, the scan's _meta.json
    (written by oct_scan.py), or by connecting the camera and reading its
    sensor info. A capture downsampled by N (oct_process --downsample, the _dsN
    file tag) multiplies the pitch by N.
  * Axial (z): depth indices are frame numbers; z = index * frame_stride *
    step_mm. step_mm comes from --step-mm or the _meta.json. frame_stride
    defaults to 2 because oct_process.py feeds every other frame
    (frames[:, :, ::2]) into compute_mean_diff. Higher index = farther from
    the camera; pass --invert-z to flip z into a "height" instead.

The sibling *_maxamp.npy (saved by oct_process.py) is used, when present, as a
per-point intensity, and --amp-percentile drops the lowest-amplitude points —
those are the pixels where the OCT envelope had no clear peak.

Usage:
    python depth_to_pointcloud.py coin_captures_ids_new_test/stack_x_ds2_depth.npy
    python depth_to_pointcloud.py depth.npy --pixel-um 3.45 --step-mm 0.001
    python depth_to_pointcloud.py depth.npy --amp-percentile 20 --stride 2
"""

import argparse
import json
import re
import struct
import numpy as np
from pathlib import Path

# oct_process.py computes on frames[:, :, ::2], so one depth index = 2 captured
# frames = 2 stage steps.
DEFAULT_FRAME_STRIDE = 2


def find_meta(depth_path: Path):
    """Locate the _meta.json saved next to the stack by oct_scan.py.

    Depth files are named <stack-stem>[_dsN]_depth.npy — strip those suffixes
    to recover the stack stem.
    """
    stem = re.sub(r"(_ds\d+)?_depth$", "", depth_path.stem)
    meta_path = depth_path.parent / f"{stem}_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text()), meta_path
    return None, meta_path


def pixel_pitch_from_camera():
    """Connect the camera briefly just to read the sensor pixel pitch."""
    from camera import Camera
    cam = Camera()
    try:
        cam.connect()
        return getattr(cam, "pixel_size_um", None)
    finally:
        cam.release()


def write_ply(path: Path, xyz: np.ndarray, intensity: np.ndarray = None):
    """Write a binary little-endian PLY. xyz is (N, 3) float; intensity (N,)
    uint8 is stored as a gray vertex color (readable by Meshlab/CloudCompare/
    Open3D)."""
    n = xyz.shape[0]
    has_c = intensity is not None
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if has_c:
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    header += ["end_header"]

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if has_c:
            rec = np.zeros(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
            rec["xyz"] = xyz.astype(np.float32)
            rec["rgb"] = np.repeat(intensity[:, None], 3, axis=1)
        else:
            rec = np.zeros(n, dtype=[("xyz", "<f4", 3)])
            rec["xyz"] = xyz.astype(np.float32)
        rec.tofile(f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a depth-index map to a metric point cloud (.ply).")
    parser.add_argument("depth", type=Path, help="path to *_depth.npy from oct_process.py")
    parser.add_argument("--pixel-um", type=float, default=None,
                        help="sensor pixel pitch in um (1:1 magnification -> lateral "
                             "scale). Default: _meta.json, then query the camera.")
    parser.add_argument("--step-mm", type=float, default=None,
                        help="stage step per captured frame in mm. "
                             "Default: _meta.json.")
    parser.add_argument("--frame-stride", type=int, default=DEFAULT_FRAME_STRIDE,
                        help="captured frames per depth index (default "
                             f"{DEFAULT_FRAME_STRIDE}: oct_process uses every "
                             "other frame)")
    parser.add_argument("--downsample", type=int, default=None,
                        help="lateral downsample factor of the depth map. "
                             "Default: parsed from the _dsN file tag, else 1.")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth pixel in x and y (default 1 = all)")
    parser.add_argument("--amp-percentile", type=float, default=0,
                        help="drop points whose max amplitude is below this "
                             "percentile of *_maxamp.npy (default 0 = keep all)")
    parser.add_argument("--invert-z", action="store_true",
                        help="flip z so larger = closer (a height map rather "
                             "than a distance map)")
    parser.add_argument("--no-camera", action="store_true",
                        help="never connect the camera for the pixel pitch")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="output .ply (default: <depth stem>_cloud.ply)")
    args = parser.parse_args()

    depth = np.load(args.depth)
    print(f"Loaded depth map: {args.depth}  shape={depth.shape}  "
          f"index range {depth.min()}..{depth.max()}")

    meta, meta_path = find_meta(args.depth)
    if meta:
        print(f"Metadata: {meta_path}")
    else:
        print(f"No metadata found ({meta_path.name} missing) — using "
              "args/defaults.")

    # ── Axial scale ──
    step_mm = args.step_mm
    if step_mm is None and meta:
        step_mm = meta.get("step_mm")
    if step_mm is None:
        raise SystemExit("Stage step unknown: pass --step-mm (oct_scan.py "
                         "used 0.001) or re-capture so _meta.json exists.")
    dz_mm = step_mm * args.frame_stride
    print(f"Axial: {step_mm:g} mm/frame x stride {args.frame_stride} "
          f"= {dz_mm:g} mm per depth index")

    # ── Lateral scale ──
    pixel_um = args.pixel_um
    if pixel_um is None and meta:
        pixel_um = meta.get("pixel_size_um")
    if pixel_um is None and not args.no_camera:
        print("Pixel pitch not in metadata — querying the camera...")
        try:
            pixel_um = pixel_pitch_from_camera()
        except Exception as e:
            print(f"Camera query failed: {e}")
    if pixel_um is None:
        raise SystemExit("Pixel pitch unknown: pass --pixel-um (sensor pitch "
                         "in um; 1:1 magnification).")
    ds = args.downsample
    if ds is None:
        m = re.search(r"_ds(\d+)_depth$", args.depth.stem)
        ds = int(m.group(1)) if m else 1
    dxy_mm = pixel_um * 1e-3 * ds
    print(f"Lateral: {pixel_um:g} um/px x downsample {ds} "
          f"= {dxy_mm:g} mm per depth-map pixel (1:1 magnification)")

    # ── Optional reliability mask from the amplitude map ──
    amp = None
    amp_path = args.depth.with_name(args.depth.stem.replace("_depth", "_maxamp")
                                    + ".npy")
    if amp_path.exists():
        amp = np.load(amp_path)
        print(f"Amplitude map: {amp_path.name}")
    elif args.amp_percentile > 0:
        raise SystemExit(f"--amp-percentile needs {amp_path.name}, not found.")

    s = max(args.stride, 1)
    depth_s = depth[::s, ::s]
    h, w = depth_s.shape
    ys, xs = np.mgrid[0:h, 0:w]

    z = depth_s.astype(np.float64) * dz_mm
    if args.invert_z:
        z = z.max() - z
    xyz = np.column_stack([
        (xs.ravel() * s * dxy_mm),
        (ys.ravel() * s * dxy_mm),
        z.ravel(),
    ])

    intensity = None
    keep = np.ones(xyz.shape[0], bool)
    if amp is not None:
        amp_s = amp[::s, ::s].ravel()
        if args.amp_percentile > 0:
            thresh = np.percentile(amp_s, args.amp_percentile)
            keep = amp_s >= thresh
            print(f"Amplitude filter: >= p{args.amp_percentile:g} "
                  f"({thresh:.4g}) keeps {keep.sum()}/{keep.size} points")
        lo, hi = np.percentile(amp_s[keep], (1, 99))
        intensity = np.clip((amp_s - lo) / max(hi - lo, 1e-12), 0, 1)
        intensity = (intensity * 255).astype(np.uint8)[keep]
    xyz = xyz[keep]

    out = args.output or args.depth.with_name(args.depth.stem + "_cloud.ply")
    write_ply(out, xyz, intensity)
    ext = xyz.max(axis=0) - xyz.min(axis=0)
    print(f"Saved: {out}  ({xyz.shape[0]} points)")
    print(f"Extent: x {ext[0]:.3f} mm  y {ext[1]:.3f} mm  z {ext[2]:.3f} mm")


if __name__ == "__main__":
    main()
