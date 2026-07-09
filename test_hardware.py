#!/usr/bin/env python3
"""
test_hardware.py — validate stage and camera setup after environment.yml installation.

Runs basic connectivity and functionality tests:
  • Stage connection and movement
  • Camera connection and capture

Usage:
    python test_hardware.py                 # auto-detect camera (AVT or IDS)
    python test_hardware.py --camera ids    # IDS only — skip the Vimba X / vmbpy check
    python test_hardware.py --camera avt    # Allied Vision only

Exit codes:
  0 = all tests passed
  1 = stage test failed
  2 = camera test failed
  3 = both failed
"""

import sys
import time
import argparse
import glob
from pathlib import Path

VIMBA_URL = "https://www.alliedvision.com/en/support/software-downloads/vimba-x-sdk/vimba-x"


def find_vmbpy_wheel():
    """Search the standard Vimba X install locations for the vmbpy wheel."""
    if sys.platform == "darwin":
        patterns = [
            "/Users/Shared/Allied Vision/**/vmbpy-*.whl",
            "/Applications/VimbaX_*/**/vmbpy-*.whl",
        ]
    elif sys.platform.startswith("linux"):
        patterns = ["/opt/**/vmbpy-*.whl"]
    elif sys.platform.startswith("win"):
        patterns = ["C:/Program Files/Allied Vision/**/vmbpy-*.whl"]
    else:
        patterns = []
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def check_vmbpy():
    """Return (ok, fix_hint). ok is True when vmbpy imports.

    When vmbpy is missing, fix_hint is the actionable install guidance — it is
    printed inline here AND repeated after the summary by main(), so the fix
    is the last thing on screen rather than scrolled away above the traceback.
    Only relevant for Allied Vision cameras — IDS users can skip with --camera ids.
    """
    try:
        import vmbpy  # noqa: F401
        return True, None
    except ImportError:
        lines = ["vmbpy (Allied Vision Vimba X SDK) is not installed."]
        wheel = find_vmbpy_wheel()
        if wheel:
            lines += ["Good news: the Vimba X SDK is installed and its wheel was found.",
                      "Install it into THIS Python and re-run:",
                      f'    ./python -m pip install "{wheel}"']
        else:
            lines += [f"Vimba X not found. Download it from:\n    {VIMBA_URL}"]
            if sys.platform == "darwin":
                lines += ["macOS installer: VimbaX_Setup-2023-4-macOS.dmg",
                          "After install, the wheel is at:",
                          "    /Users/Shared/Allied Vision/Vimba X/Vmbpy/vmbpy-*.whl"]
            elif sys.platform.startswith("linux"):
                lines += ["Linux installer: VimbaX_Setup-2026-1-Linux64.tar.gz (or _ARM64)",
                          "Wheel: /opt/VimbaX_<version>/api/python/vmbpy-*.whl"]
            elif sys.platform.startswith("win"):
                lines += ["Windows installer: VimbaX_Setup-2026-1-Win64.exe",
                          "Wheel: C:\\Program Files\\Allied Vision\\Vimba X\\api\\python\\vmbpy-*.whl"]
            lines += ["Then: ./python -m pip install <path-to-vmbpy-*.whl>"]
        lines += ["(If you are using an IDS camera, re-run with: --camera ids)"]
        hint = "\n".join(lines)
        print("⚠ " + hint.replace("\n", "\n  "))
        return False, hint


def test_stage():
    """Test stage connection and movement."""
    print("\n" + "="*60)
    print("  TEST: Stage (Thorlabs KDC101)")
    print("="*60)

    try:
        from stage import ThorlabsStage

        print("✓ Imported ThorlabsStage")

        stage = ThorlabsStage(units="mm")
        print("✓ Created ThorlabsStage instance")

        stage.connect()
        print("✓ Connected to stage")

        initial_pos = stage.position
        print(f"✓ Read initial position: {initial_pos:.3f} mm")

        # Move by a small amount (5 mm)
        target = initial_pos + 5.0
        stage.move_to(target)
        print(f"✓ Moved to {target:.3f} mm")

        time.sleep(0.5)  # Let motion complete
        actual = stage.position
        error = abs(actual - target)

        if error > 0.1:
            print(f"⚠ Position mismatch: target={target:.3f}, actual={actual:.3f}, error={error:.3f} mm")
        else:
            print(f"✓ Position verified: {actual:.3f} mm (error: {error:.3f} mm)")

        # Move back
        stage.move_to(initial_pos)
        print(f"✓ Moved back to initial position")

        stage.release()
        print("✓ Released stage connection")

        print("\n✓✓✓ Stage test PASSED")
        return True

    except Exception as e:
        print(f"\n✗✗✗ Stage test FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        return False


def test_camera(vendor="auto"):
    """Test camera connection and capture.

    vendor: "auto" (AVT then IDS), "avt" (Allied Vision), or "ids".
    For "avt"/"auto" the Vimba X / vmbpy install is checked first; "ids" skips it.
    """
    label = {"auto": "Allied Vision or IDS", "avt": "Allied Vision", "ids": "IDS"}[vendor]
    print("\n" + "="*60)
    print(f"  TEST: Camera ({label})")
    print("="*60)

    # vmbpy is only needed for Allied Vision cameras. Skip the check for IDS.
    hint = None
    if vendor in ("auto", "avt"):
        has_vmbpy, hint = check_vmbpy()
        if vendor == "avt" and not has_vmbpy:
            print("\n✗✗✗ Camera test FAILED — vmbpy not installed (see above).")
            return False, hint

    try:
        from camera import Camera
        import numpy as np

        print("✓ Imported Camera")

        # Create test save directory
        test_dir = Path("test_captures")
        test_dir.mkdir(exist_ok=True)
        print(f"✓ Created test directory: {test_dir}")

        prefer = None if vendor == "auto" else vendor
        cam = Camera(exposure_us=10000, gain_db=0.0, save_dir=str(test_dir), prefer=prefer)
        print("✓ Created Camera instance")

        cam.connect()
        print("✓ Connected to camera")

        # Single capture
        frame = cam.capture()
        print(f"✓ Captured frame: shape={frame.shape}, dtype={frame.dtype}")

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected ndarray, got {type(frame)}")

        if frame.size == 0:
            raise ValueError("Captured frame is empty")

        # Multi-frame capture to buffer
        cam.capture_to_buffer()
        cam.capture_to_buffer()
        cam.capture_to_buffer()
        print(f"✓ Captured 3 frames to buffer")

        # Save stack (just filename, not full path — camera handles save_dir)
        cam.save_stack("test_stack.npy", save_previews=False)
        stack_file = test_dir / "test_stack.npy"
        if stack_file.exists():
            size_mb = stack_file.stat().st_size / (1024 * 1024)
            print(f"✓ Saved stack: {stack_file} ({size_mb:.1f} MB)")
        else:
            raise FileNotFoundError(f"Stack file not created: {stack_file}")

        cam.release()
        print("✓ Released camera connection")

        print("\n✓✓✓ Camera test PASSED")
        return True, None

    except Exception as e:
        print(f"\n✗✗✗ Camera test FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False, hint


def main():
    parser = argparse.ArgumentParser(description="Validate stage and camera setup.")
    parser.add_argument("--camera", choices=["auto", "avt", "ids"], default="auto",
                        help="which camera vendor to test (default auto; "
                             "'ids' skips the Vimba X / vmbpy check)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  HARDWARE VALIDATION")
    print("="*60)
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")
    print(f"  Camera vendor: {args.camera}")

    stage_ok = test_stage()
    camera_ok, camera_hint = test_camera(vendor=args.camera)

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Stage:  {'✓ PASS' if stage_ok else '✗ FAIL'}")
    print(f"  Camera: {'✓ PASS' if camera_ok else '✗ FAIL'}")
    print("="*60)

    # Repeat the actionable camera fix LAST, so it's the first thing people
    # see at the bottom of the terminal instead of scrolled away above.
    if not camera_ok and camera_hint:
        print("\n  ▶ HOW TO FIX THE CAMERA:")
        print("  " + camera_hint.replace("\n", "\n  "))

    # Exit code: bitfield (bit 0 = stage, bit 1 = camera)
    exit_code = (0 if stage_ok else 1) + (0 if camera_ok else 2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
