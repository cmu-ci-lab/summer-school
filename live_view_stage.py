#!/usr/bin/env python3
"""
live_view_stage.py — Real-time camera viewer with exposure AND stage control.

Like live_view.py, but if a Thorlabs stage is found it also lets you drive it
from the keyboard to emulate a focus sweep. If no stage is found, it silently
falls back to a camera-only live view.

Controls:
  +/-        : Increase/decrease exposure
  up / w     : Move stage by +coarse step (hold to sweep)
  down / s   : Move stage by -coarse step (hold to sweep)
  e / d      : Move stage by +/- fine step (coarse / FINE_RATIO)
  [ / ]      : Decrease / increase the coarse step size
  q          : Quit

The coarse step (mm per key event) is set with --step; the fine step (e/d) is
1/FINE_RATIO of it. Holding a move key repeats it at the OS key-repeat rate,
sweeping the stage while the preview keeps running.

Usage:
    python live_view_stage.py                  # default 0.05 mm step
    python live_view_stage.py --step 0.1       # 0.1 mm per key event
    python live_view_stage.py --binning 4 --exposure 10000
"""

import cv2
import argparse
import numpy as np
from camera import Camera
import time

# Arrow-key codes vary by OS/GUI backend; match against all known up/down values
# (macOS Qt, GTK, Windows). w/s are provided as backend-independent alternates.
UP_KEYS   = {63232, 65362, 2490368, 82}
DOWN_KEYS = {63233, 65364, 2621440, 84}

# e/d move a finer step, FINE_RATIO times smaller than the w/s coarse step.
FINE_RATIO = 10


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
        return stage
    except Exception as e:
        print(f"No stage found ({e}); running camera-only.")
        return None


def main():
    parser = argparse.ArgumentParser(description="Live camera viewer with stage control.")
    parser.add_argument("--step", type=float, default=1.0,
                        help="coarse stage step in mm (w/s or arrows); e/d "
                             f"moves 1/{FINE_RATIO} of this (default 1.0)")
    parser.add_argument("--binning", type=int, default=4,
                        help="on-sensor binning factor for the preview (default 4)")
    parser.add_argument("--exposure", type=float, default=10000,
                        help="initial exposure in microseconds, non-binned-equivalent (default 10000)")
    parser.add_argument("--gamma", type=float, default=2.2,
                        help="display gamma; >1 brightens dark scenes (default 2.2, 1.0 = linear)")
    args = parser.parse_args()
    gamma = args.gamma

    print("Initializing camera...")
    cam = Camera(exposure_us=args.exposure, gain_db=0.0, save_dir="live_view_captures")
    cam.connect()

    stage = connect_stage()
    coarse_mm = args.step   # w/s or arrows; Shift (W/S) moves coarse_mm / FINE_RATIO

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
        """Move the stage (non-blocking so the preview keeps running).

        Coarse = coarse_mm; fine (Shift) = coarse_mm / FINE_RATIO.
        """
        if stage is None:
            return
        dist = coarse_mm / FINE_RATIO if fine else coarse_mm
        try:
            stage.move_by(direction * dist, wait=False)
        except Exception as e:
            print(f"Stage move failed: {e}")

    # Size the window to fit the screen (leave a margin for window chrome).
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    window_name = "Live View + Stage"
    # WINDOW_GUI_NORMAL disables OpenCV's Qt toolbar. The toolbar's icon engine
    # (QAppleIconEngine) crashes on Qt 6.11 + macOS 12 with a doesNotRecognizeSelector
    # NSException, so we must not create it. WINDOW_NORMAL keeps the window resizable.
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    window_sized = False

    # Stream continuously when the camera supports it (avoids per-frame
    # start/stop overhead). Otherwise fall back to single-shot capture().
    streaming = hasattr(cam, "start_streaming")
    if streaming:
        cam.start_streaming(buffer_count=5)

    def grab():
        if streaming:
            return cam.latest_frame()
        return cam.capture()

    print("Starting live view. +/- exposure | w/s (or arrows) move stage, e/d = fine, [ ] step | q quit.")
    print(f"Initial exposure: {exposure_us} µs (non-binned equivalent)  |  "
          f"coarse step: {coarse_mm:g} mm, fine: {coarse_mm / FINE_RATIO:g} mm")

    # FPS measured over a window. When streaming, count real frames delivered by
    # the camera (the loop itself runs faster than frames arrive).
    fps = 0.0
    fps_t0 = time.time()
    fps_count0 = cam.frame_count if streaming else 0

    # Stage position is polled at ~10 Hz (not every frame) to limit serial traffic.
    pos_mm = None
    moving = False
    last_pos_t = 0.0

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

            # Scale down to fit the screen while preserving aspect ratio
            h, w = frame_rgb.shape[:2]
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            # Measure actual frame rate over a 0.5 s window
            now = time.time()
            if streaming:
                if now - fps_t0 >= 0.5:
                    fps = (cam.frame_count - fps_count0) / (now - fps_t0)
                    fps_t0, fps_count0 = now, cam.frame_count
            else:
                loop_time = now - loop_start
                fps = 1.0 / loop_time if loop_time > 0 else 0.0

            # Poll stage position at ~10 Hz
            if stage is not None and now - last_pos_t > 0.1:
                try:
                    pos_mm = stage.position
                    moving = stage.is_moving
                except Exception:
                    pass
                last_pos_t = now

            # Text overlay
            text = f"Exposure: {int(exposure_us)} us | gamma {gamma:g} | {fps:0.1f} FPS"
            cv2.putText(frame_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            controls = f"+/- exposure   q quit   (binning {live_binning}x)"
            cv2.putText(frame_rgb, controls, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
            if stage is not None:
                pos_str = f"{pos_mm:.4f}" if pos_mm is not None else "?"
                stage_text = (f"Stage: {pos_str} mm   w/s {coarse_mm:g}  e/d {coarse_mm / FINE_RATIO:g} mm"
                              f"   [ ] step{'  MOVING' if moving else ''}")
            else:
                stage_text = "Stage: not found (camera only)"
            cv2.putText(frame_rgb, stage_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 200, 255), 2)

            # Display
            cv2.imshow(window_name, frame_rgb)
            if not window_sized:
                cv2.resizeWindow(window_name, frame_rgb.shape[1], frame_rgb.shape[0])
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
