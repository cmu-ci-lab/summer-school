#!/usr/bin/env python3
"""
oct_crop_scan.py — coherence-envelope scan over the full stage travel using a
cropped camera ROI.

Crops the sensor readout to a small region (fast capture), sweeps the stage
across its whole travel (default 0 → 25 mm in 10 µm steps ≈ 2501 frames),
runs compute_mean_diff on the cropped stack, and plots the interference
amplitude vs stage position with a fitted Gaussian envelope — peak position
and FWHM annotated. Everything is saved: the curve (.npz), the plot (.png),
and optionally the raw cropped stack.

Two scan modes:
  stepped     move -> settle -> capture at every step (exact positions; slow —
              ~0.25 s of stage overhead per step, so 2501 steps ≈ 10 min)
  continuous  ONE continuous move at --velocity-mm-s while the camera
              free-runs into a buffer; each frame's position is interpolated
              from a timestamped position log (fast — travel/velocity seconds;
              frame spacing = velocity / camera fps)

Usage:
    python oct_crop_scan.py                          # stepped, 128x128 centre crop
    python oct_crop_scan.py --mode continuous --velocity-mm-s 0.5
    python oct_crop_scan.py --crop 900 700 128 128   # explicit x y w h
    python oct_crop_scan.py --start 10 --end 15 --step-mm 0.005
    python oct_crop_scan.py --from-file oct_crop_scans/cropscan_x.npz   # re-plot
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from oct import compute_mean_diff


# ──────────────────────────────────────────────────────────────────────────────
# Analysis: envelope measurement + Gaussian fit
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_floor(z, amp, z0, sigma, floor):
    return floor + amp * np.exp(-0.5 * ((z - z0) / sigma) ** 2)


def fit_envelope(positions, amplitude):
    """Fit floor + Gaussian to the amplitude curve.

    Returns a dict with peak_mm, fwhm_mm, fit params and the direct
    (interpolated half-max crossing) FWHM as a sanity check — or None when
    there's no usable peak / the fit fails.
    """
    from scipy.optimize import curve_fit
    z, a = np.asarray(positions, float), np.asarray(amplitude, float)
    floor0 = float(np.percentile(a, 10))
    ipk = int(np.argmax(a))
    amp0 = float(a[ipk] - floor0)
    if amp0 <= 0:
        return None
    # Initial sigma: distance from the peak to the half-max crossing.
    half = floor0 + amp0 / 2
    above = np.nonzero(a > half)[0]
    sigma0 = max((z[above[-1]] - z[above[0]]) / 2.355, np.mean(np.diff(z))) \
        if len(above) >= 2 else np.mean(np.diff(z)) * 3

    try:
        popt, _ = curve_fit(gaussian_floor, z, a,
                            p0=[amp0, z[ipk], sigma0, floor0],
                            maxfev=10000)
    except Exception:
        return None
    amp, z0, sigma, floor = popt
    sigma = abs(sigma)
    if amp <= 0 or not (z.min() - 1 <= z0 <= z.max() + 1):
        return None

    # Direct FWHM from interpolated half-max crossings (fit-independent).
    direct = None
    if len(above) >= 2:
        i0, i1 = above[0], above[-1]
        left = z[i0] if i0 == 0 else np.interp(half, [a[i0 - 1], a[i0]],
                                               [z[i0 - 1], z[i0]])
        right = z[i1] if i1 == len(z) - 1 else np.interp(
            half, [a[i1 + 1], a[i1]], [z[i1 + 1], z[i1]])
        direct = float(right - left)

    return {"peak_mm": float(z0), "fwhm_mm": float(2.3548 * sigma),
            "amp": float(amp), "sigma_mm": float(sigma),
            "floor": float(floor), "direct_fwhm_mm": direct,
            "n_above_half": int(len(above))}


def warn_if_undersampled(fit, step_mm):
    """A coarse sweep locates the peak well but can't resolve the width."""
    if fit and fit["n_above_half"] < 6:
        print(f"NOTE: only {fit['n_above_half']} samples above half-max — the "
              "envelope is under-sampled at this step size, so treat the FWHM "
              "as indicative only (the peak position is still good). For a "
              "reliable FWHM, re-scan a narrow range with finer steps, e.g.:\n"
              f"  python oct_crop_scan.py --start {fit['peak_mm'] - 0.15:.3f} "
              f"--end {fit['peak_mm'] + 0.15:.3f} --step-mm 0.002")


def plot_envelope(positions, amplitude, fit, out_png, title="", show=True,
                  zoom_mm=0.1):
    """Amplitude vs position, with a second panel zoomed to peak ± zoom_mm.

    Top: the full sweep with the fitted envelope. Bottom (when a peak was
    found): the ±zoom_mm neighbourhood of the peak with individual samples
    visible — the panel that tells you whether the envelope is actually
    resolved or just a one-sample spike.
    """
    import matplotlib.pyplot as plt
    z, a = np.asarray(positions, float), np.asarray(amplitude, float)

    if fit is not None:
        fig, (ax, axz) = plt.subplots(
            2, 1, figsize=(10, 8), height_ratios=[3, 2])
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        axz = None

    ax.plot(z, a, ".", ms=2.5, color="#3987e5", alpha=0.55,
            label="measured amplitude")
    if fit is not None:
        zf = np.linspace(z.min(), z.max(), 4000)
        ax.plot(zf, gaussian_floor(zf, fit["amp"], fit["peak_mm"],
                                   fit["sigma_mm"], fit["floor"]),
                color="#9085e9", lw=1.8, label="Gaussian envelope fit")
        ax.axvline(fit["peak_mm"], color="#9085e9", ls="--", lw=1, alpha=0.7)
        direct = (f"\ndirect FWHM: {fit['direct_fwhm_mm'] * 1000:.1f} µm"
                  if fit.get("direct_fwhm_mm") else "")
        ax.annotate(
            f"peak: {fit['peak_mm']:.4f} mm\n"
            f"fit FWHM: {fit['fwhm_mm'] * 1000:.1f} µm" + direct,
            xy=(fit["peak_mm"], fit["floor"] + fit["amp"]),
            xytext=(12, -8), textcoords="offset points",
            fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#9085e9",
                      alpha=0.9))
    ax.set_ylabel("interference amplitude (mean |I − DC|)")
    ax.set_title(title or "Coherence envelope")
    ax.legend(loc="upper right", frameon=False)
    ax.margins(x=0.02)

    # ── Zoom panel: peak ± zoom_mm ──
    if axz is not None:
        z0 = fit["peak_mm"]
        m = np.abs(z - z0) <= zoom_mm
        if m.sum() >= 2:
            zf = np.linspace(z0 - zoom_mm, z0 + zoom_mm, 2000)
            axz.plot(zf, gaussian_floor(zf, fit["amp"], z0, fit["sigma_mm"],
                                        fit["floor"]),
                     color="#9085e9", lw=1.8)
            axz.plot(z[m], a[m], "o-", ms=5, lw=0.8, color="#3987e5",
                     alpha=0.85)
            half = fit["floor"] + fit["amp"] / 2
            axz.axhline(half, color="#c98500", ls=":", lw=1)
            axz.text(z0 - zoom_mm * 0.98, half, " half-max", fontsize=9,
                     color="#c98500", va="bottom")
            axz.axvline(z0, color="#9085e9", ls="--", lw=1, alpha=0.7)
            axz.set_xlim(z0 - zoom_mm, z0 + zoom_mm)
            axz.set_title(f"zoom: peak ± {zoom_mm * 1000:.0f} µm "
                          f"({int(m.sum())} samples)", fontsize=10)
        else:
            axz.text(0.5, 0.5, "no samples within the zoom window",
                     ha="center", va="center", transform=axz.transAxes)
        axz.set_xlabel("stage position (mm)")
        axz.set_ylabel("amplitude")
    else:
        ax.set_xlabel("stage position (mm)")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot: {out_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def process(stack, patch_size, temporal_window):
    """Cropped (H, W, N) stack -> per-frame interference amplitude (N,)."""
    md_vector, _, _ = compute_mean_diff(stack, patch_size=patch_size,
                                        avg_type="local",
                                        temporal_window=temporal_window)
    return md_vector


# ──────────────────────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────────────────────

def sweep_stepped(stage, cam, positions):
    """move -> settle -> capture at every position. Exact but slow."""
    frames = []
    n = len(positions)
    t0 = time.time()
    for i, pos in enumerate(positions):
        stage.move_to(pos)
        frames.append(cam.capture())
        if (i + 1) % 100 == 0 or i == n - 1:
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            print(f"  [{i + 1}/{n}]  {pos:.3f} mm   "
                  f"elapsed {el / 60:.1f} min, eta {eta / 60:.1f} min")
    return np.asarray(positions, float), frames


def sweep_continuous(stage, cam, start, end, velocity_mm_s,
                     max_frames=60000, poll_every=5):
    """ONE continuous move while the camera free-runs into a buffer.

    Frames are timestamped as they arrive; the stage position is polled (with
    timestamps) every `poll_every` frames; afterwards each frame's position is
    interpolated from the position log. Frame spacing = velocity / fps.
    """
    print(f"Moving to start ({start:g} mm)...")
    stage.move_to(start)

    old_v = None
    try:
        old_v = stage.get_velocity()
        stage.set_velocity(velocity_mm_s)
        print(f"Sweep velocity: {velocity_mm_s:g} mm/s "
              f"(was {old_v:g}) — sweep ≈ {(end - start) / velocity_mm_s:.0f} s")
    except Exception as e:
        print(f"Could not set stage velocity ({e}); sweeping at the current "
              "velocity — frame spacing will follow it.")

    frames, frame_t = [], []
    pos_t, pos_v = [], []

    def poll():
        ta = time.time()
        v = stage.position
        pos_t.append((ta + time.time()) / 2)   # midpoint of the serial query
        pos_v.append(v)

    try:
        poll()
        stage.move_to(end, wait=False)
        t0 = time.time()
        while len(frames) < max_frames:
            frame_t.append(time.time())
            frames.append(cam.capture())
            if len(frames) % poll_every == 0:
                poll()
                if not stage.is_moving and abs(pos_v[-1] - end) < 0.05:
                    break
                if len(frames) % (poll_every * 40) == 0:
                    print(f"  {pos_v[-1]:.3f} mm   {len(frames)} frames   "
                          f"{len(frames) / (time.time() - t0):.0f} fps")
        poll()
    finally:
        if old_v is not None:
            try:
                stage.set_velocity(old_v)
            except Exception:
                pass

    positions = np.interp(frame_t, pos_t, pos_v)
    spacing = np.median(np.diff(positions[np.argsort(frame_t)]))
    print(f"Captured {len(frames)} frames; median spacing "
          f"{spacing * 1000:.1f} µm ({len(pos_v)} position samples).")
    # Sort by position so downstream fitting/plotting sees a monotone axis.
    order = np.argsort(positions, kind="stable")
    positions = positions[order]
    frames = [frames[i] for i in order]
    return positions, frames


def main():
    p = argparse.ArgumentParser(
        description="Cropped-ROI coherence-envelope scan over the stage travel.")
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"),
                   default=None,
                   help="sensor ROI; default: a --crop-size box at the centre")
    p.add_argument("--crop-size", type=int, nargs=2, metavar=("W", "H"),
                   default=(128, 128),
                   help="ROI size when --crop is not given (default 128 128; "
                        "keep it small — processing memory grows as W*H*frames)")
    p.add_argument("--start", type=float, default=0.0,
                   help="sweep start in mm (default 0)")
    p.add_argument("--end", type=float, default=25.0,
                   help="sweep end in mm (default 25)")
    p.add_argument("--step-mm", type=float, default=0.010,
                   help="stepped mode: step size in mm (default 0.010 = 10 um "
                        "-> 2501 frames)")
    p.add_argument("--mode", choices=["stepped", "continuous"], default="stepped",
                   help="stepped (exact positions, slow) or continuous (one "
                        "move, frames buffered + positions interpolated, fast)")
    p.add_argument("--velocity-mm-s", type=float, default=0.5,
                   help="continuous mode: sweep velocity in mm/s (default 0.5; "
                        "frame spacing = velocity / camera fps)")
    p.add_argument("--exposure", type=float, default=None,
                   help="exposure in us (default: last visualizer session, "
                        "else the sensor's current value)")
    p.add_argument("--gain", type=float, default=0.0, help="camera gain")
    p.add_argument("--camera", choices=["auto", "avt", "ids"], default="auto")
    p.add_argument("--save-dir", default="oct_crop_scans")
    p.add_argument("--no-home", action="store_true", help="skip homing first")
    p.add_argument("--save-stack", action="store_true",
                   help="also save the raw cropped stack into the .npz")
    p.add_argument("--patch-size", type=int, default=5,
                   help="compute_mean_diff spatial patch (default 5)")
    p.add_argument("--temporal-window", type=int, default=20,
                   help="compute_mean_diff DC window in frames (default 20)")
    p.add_argument("--from-file", type=Path, default=None,
                   help="re-process/re-plot a saved cropscan_*.npz instead of "
                        "capturing (no hardware needed)")
    p.add_argument("--zoom-um", type=float, default=100,
                   help="half-width of the zoomed peak panel in um (default 100)")
    p.add_argument("--no-show", action="store_true",
                   help="save the plot without opening a window")
    args = p.parse_args()

    # ── Re-process a saved scan: no hardware path ──
    if args.from_file:
        data = np.load(args.from_file, allow_pickle=False)
        positions = data["positions"]
        if "stack" in data:
            amplitude = process(data["stack"].astype(np.float64),
                                args.patch_size, args.temporal_window)
        else:
            amplitude = data["amplitude"]
        fit = fit_envelope(positions, amplitude)
        step = float(np.median(np.diff(positions))) if len(positions) > 1 else 0
        warn_if_undersampled(fit, step)
        out_png = args.from_file.with_suffix(".png")
        plot_envelope(positions, amplitude, fit, out_png,
                      title=args.from_file.stem, show=not args.no_show,
                      zoom_mm=args.zoom_um / 1000)
        if fit:
            print(f"Peak {fit['peak_mm']:.4f} mm   "
                  f"FWHM {fit['fwhm_mm'] * 1000:.1f} µm (fit)"
                  + (f" / {fit['direct_fwhm_mm'] * 1000:.1f} µm (direct)"
                     if fit.get("direct_fwhm_mm") else ""))
        return

    if args.mode == "stepped":
        targets = np.arange(args.start, args.end + args.step_mm / 2, args.step_mm)
        est_min = len(targets) * 0.25 / 60   # ~0.25 s/step: move + settle
        print(f"Stepped sweep: {len(targets)} steps of "
              f"{args.step_mm * 1000:g} µm ({args.start:g} → {args.end:g} mm) "
              f"— roughly {est_min:.0f} min.")
        print("  (Time is dominated by the per-step stage move/settle, not "
              "the camera: a cropped frame reads out in ~1 ms. Frames are "
              "buffered in memory; all processing happens after the sweep.)")
    else:
        print(f"Continuous sweep: {args.start:g} → {args.end:g} mm at "
              f"{args.velocity_mm_s:g} mm/s ≈ "
              f"{(args.end - args.start) / args.velocity_mm_s:.0f} s; frame "
              "positions interpolated from a timestamped position log.")

    # Exposure: CLI > last visualizer session > sensor current (see oct_scan).
    exposure = args.exposure
    if exposure is None:
        from exposure_store import load_exposure
        stored = load_exposure()
        if stored is not None:
            exposure, age = stored
            print(f"Using exposure from your last visualizer session: "
                  f"{exposure:g} µs  (saved {age})")

    from stage import ThorlabsStage
    from camera import Camera

    stage = ThorlabsStage(units="mm")
    stage.connect()
    cam = Camera(exposure_us=exposure, gain_db=args.gain,
                 save_dir=args.save_dir,
                 prefer=None if args.camera == "auto" else args.camera)
    cam.connect()

    try:
        # ── Crop the sensor: centre box unless --crop was explicit ──
        full = cam.capture()
        fh, fw = full.shape[:2]
        if args.crop is not None:
            x, y, w, h = args.crop
        else:
            w, h = args.crop_size
            x, y = (fw - w) // 2, (fh - h) // 2
        x, y, w, h = cam.set_roi(x, y, w, h)

        if not args.no_home:
            stage.home()

        if args.mode == "stepped":
            positions, frames = sweep_stepped(stage, cam, targets)
        else:
            positions, frames = sweep_continuous(stage, cam, args.start,
                                                 args.end, args.velocity_mm_s)
        n = len(positions)

        stack = np.stack(frames, axis=-1)
        print(f"Captured stack: {stack.shape}  {stack.dtype}")

        print("Running compute_mean_diff...")
        amplitude = process(stack.astype(np.float64),
                            args.patch_size, args.temporal_window)

        # ── Save curve (+ optional stack) and metadata ──
        out_dir = Path(cam.save_dir)
        stem = f"cropscan_{time.strftime('%Y%m%d_%H%M%S')}"
        eff_step = (float(np.median(np.diff(positions))) if n > 1
                    else args.step_mm)
        meta = {"mode": args.mode,
                "crop_xywh": [int(x), int(y), int(w), int(h)],
                "start_mm": args.start, "end_mm": args.end,
                "step_mm": eff_step, "n_frames": n,
                "velocity_mm_s": (args.velocity_mm_s
                                  if args.mode == "continuous" else None),
                "exposure_us": cam.exposure_us,
                "patch_size": args.patch_size,
                "temporal_window": args.temporal_window,
                "pixel_size_um": getattr(cam, "pixel_size_um", None)}
        arrays = dict(positions=positions, amplitude=amplitude,
                      meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
        if args.save_stack:
            arrays["stack"] = stack
        out_npz = out_dir / f"{stem}.npz"
        np.savez_compressed(out_npz, **arrays)
        print(f"Saved: {out_npz}")

        # ── Fit + plot ──
        fit = fit_envelope(positions, amplitude)
        if fit:
            print(f"Peak {fit['peak_mm']:.4f} mm   "
                  f"FWHM {fit['fwhm_mm'] * 1000:.1f} µm (fit)"
                  + (f" / {fit['direct_fwhm_mm'] * 1000:.1f} µm (direct)"
                     if fit.get("direct_fwhm_mm") else ""))
        else:
            print("No usable envelope peak found — plotting raw curve only.")
        warn_if_undersampled(fit, eff_step)
        plot_envelope(positions, amplitude, fit, out_dir / f"{stem}.png",
                      title=f"{stem}   crop {w}x{h} @ ({x},{y})",
                      show=not args.no_show, zoom_mm=args.zoom_um / 1000)
    finally:
        stage.release()
        cam.release()


if __name__ == "__main__":
    main()
