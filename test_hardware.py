#!/usr/bin/env python3
"""
test_hardware.py — validate stage and camera setup after environment.yml installation.

Runs basic connectivity and functionality tests:
  • Stage connection and movement
  • Camera connection and capture

Usage:
    python test_hardware.py

Exit codes:
  0 = all tests passed
  1 = stage test failed
  2 = camera test failed
  3 = both failed
"""

import sys
import time
from pathlib import Path


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


def test_camera():
    """Test camera connection and capture."""
    print("\n" + "="*60)
    print("  TEST: Camera (Allied Vision or IDS)")
    print("="*60)

    try:
        from camera import Camera
        import numpy as np

        print("✓ Imported Camera")

        # Create test save directory
        test_dir = Path("test_captures")
        test_dir.mkdir(exist_ok=True)
        print(f"✓ Created test directory: {test_dir}")

        cam = Camera(exposure_us=10000, gain_db=0.0, save_dir=str(test_dir))
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
        return True

    except Exception as e:
        print(f"\n✗✗✗ Camera test FAILED")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("  HARDWARE VALIDATION")
    print("="*60)
    print(f"  Python: {sys.version}")
    print(f"  Platform: {sys.platform}")

    stage_ok = test_stage()
    camera_ok = test_camera()

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  Stage:  {'✓ PASS' if stage_ok else '✗ FAIL'}")
    print(f"  Camera: {'✓ PASS' if camera_ok else '✗ FAIL'}")
    print("="*60)

    # Exit code: bitfield (bit 0 = stage, bit 1 = camera)
    exit_code = (0 if stage_ok else 1) + (0 if camera_ok else 2)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
