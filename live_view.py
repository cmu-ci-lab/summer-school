#!/usr/bin/env python3
"""
live_view.py — Real-time camera viewer with exposure control.

Controls:
  +/-       : Increase/decrease exposure
  q         : Quit

Uses continuous streaming and 4x4 on-sensor binning for a fast preview.
The exposure shown is the *non-binned-equivalent* value: internally it is
divided by binning**2 so the preview brightness matches what a full-resolution
capture at the displayed exposure would look like.

Usage:
    python live_view.py
"""

import cv2
import numpy as np
from camera import Camera
import time


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


def main():
    print("Initializing camera...")
    cam = Camera(exposure_us=10000, gain_db=0.0, save_dir="live_view_captures")
    cam.connect()

    exposure_us = cam.exposure_us
    exposure_step = 500  # microseconds
    frame_time_target = 1.0 / 30.0  # 30 FPS

    # Bin on-sensor for the live preview. 4x4 binning sums 16 pixels, so the
    # image is ~16x brighter; we divide the actual exposure by binning**2 so the
    # exposure *shown* equals the non-binned-equivalent value usable elsewhere.
    live_binning = 4
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

    # Size the window to fit the screen (leave a margin for window chrome).
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)

    window_name = "Live View"
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

    print("Starting live view. Press +/- to adjust exposure, q to quit.")
    print(f"Initial exposure: {exposure_us} µs (non-binned equivalent)")

    # FPS measured over a window. When streaming, count real frames delivered by
    # the camera (the loop itself runs faster than frames arrive).
    fps = 0.0
    fps_t0 = time.time()
    fps_count0 = cam.frame_count if streaming else 0

    try:
        while True:
            loop_start = time.time()

            # Grab the latest frame; may be None briefly while streaming spins up
            frame = grab()
            if frame is None:
                cv2.waitKey(1)
                continue

            # Convert image to 8-bit for display
            if frame.dtype == np.uint16:
                max_val = int(frame.max()) if frame.size else 0
                if max_val <= 4095:
                    frame_display = (frame >> 4).astype(np.uint8)
                else:
                    frame_display = (frame >> 8).astype(np.uint8)
            elif frame.dtype == np.uint32 or frame.dtype == np.int32:
                frame_display = (frame.astype(np.float32) / frame.max() * 255).clip(0, 255).astype(np.uint8)
            else:
                frame_display = frame.astype(np.uint8)
            
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

            # Add text overlay
            text = f"Exposure: {int(exposure_us)} us | {fps:0.1f} FPS"
            cv2.putText(frame_rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            controls = f"+/- : exposure    q : quit    (binning {live_binning}x)"
            cv2.putText(frame_rgb, controls, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 255), 2)
            
            # Display
            cv2.imshow(window_name, frame_rgb)
            if not window_sized:
                cv2.resizeWindow(window_name, frame_rgb.shape[1], frame_rgb.shape[0])
                window_sized = True
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting...")
                break
            elif key == ord('+') or key == ord('='):
                exposure_us = min(exposure_us + exposure_step, 1000000)  # Cap at 1s
                apply_exposure(exposure_us)
                print(f"Exposure: {exposure_us} µs")
            elif key == ord('-') or key == ord('_'):
                exposure_us = max(exposure_us - exposure_step, 100)  # Floor at 100 µs
                apply_exposure(exposure_us)
                print(f"Exposure: {exposure_us} µs")
            
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
        cv2.destroyAllWindows()
        print("Camera released.")


if __name__ == "__main__":
    main()
