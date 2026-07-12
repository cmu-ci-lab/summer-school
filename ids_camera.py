import ctypes
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime

try:
    from pyueye import ueye
    _UEYE_AVAILABLE = True
except ImportError:
    _UEYE_AVAILABLE = False


class IDSCamera:
    """IDS uEye (uc480) camera wrapper using pyueye.

    Requires:
        - IDS Software Suite (uEye SDK) installed from ids-imaging.com
        - pip install pyueye

    Usage::
        with IDSCamera(exposure_us=15000) as cam:
            image = cam.capture()

    Specify serial= to target a particular camera when multiple are connected::
        with IDSCamera(serial=1) as cam: ...
    """

    def __init__(self, exposure_us: float = 10000, gain_db: float = 0.0,
                 save_dir: str = "captures", serial: int | None = None):
        if not _UEYE_AVAILABLE:
            raise ImportError(
                "pyueye not found.\n"
                "Install with: pip install pyueye\n"
                "Also requires IDS Software Suite from ids-imaging.com"
            )
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.save_dir = Path(save_dir)
        self.serial = serial
        self._hcam = None
        self._mem_ptr = None
        self._mem_id = None
        self._width = None
        self._height = None
        self._bits_per_pixel = 8
        self._dtype = np.uint8
        self._buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        cam_id = self.serial if self.serial is not None else 0
        self._hcam = ueye.HIDS(cam_id)

        ret = ueye.is_InitCamera(self._hcam, None)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_InitCamera failed (code {ret}). "
                               "Check USB connection and IDS Software Suite install.")

        sensor = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self._hcam, sensor)
        self._width  = int(sensor.nMaxWidth)
        self._height = int(sensor.nMaxHeight)
        name = sensor.strSensorName.decode()
        # Sensor pixel pitch: SENSORINFO.wPixelSize is in units of 0.01 µm.
        try:
            self.pixel_size_um = float(int(sensor.wPixelSize)) / 100.0 or None
        except Exception:
            self.pixel_size_um = None
        pitch = f"  pitch {self.pixel_size_um:g} um" if self.pixel_size_um else ""
        print(f"Connected [IDS uEye]: {name}  {self._width}x{self._height}{pitch}")

        # Try monochrome modes from highest to lowest native bit depth
        _MONO_MODES = [
            (ueye.IS_CM_MONO12, 16, np.uint16),
            (ueye.IS_CM_MONO10, 16, np.uint16),
            (ueye.IS_CM_MONO8,   8, np.uint8),
        ]
        for mode, bpp, dtype in _MONO_MODES:
            if ueye.is_SetColorMode(self._hcam, mode) == ueye.IS_SUCCESS:
                self._bits_per_pixel = bpp
                self._dtype = dtype
                print(f"Color mode: MONO{bpp if bpp == 8 else bpp} ({dtype.__name__})")
                break

        self._mem_ptr = ueye.c_mem_p()
        self._mem_id  = ueye.int()
        ueye.is_AllocImageMem(self._hcam, self._width, self._height, self._bits_per_pixel,
                              self._mem_ptr, self._mem_id)
        ueye.is_SetImageMem(self._hcam, self._mem_ptr, self._mem_id)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {self.save_dir.resolve()}")

        self._apply_settings()
        return self

    def release(self):
        if self._hcam is None:
            return
        try:
            ueye.is_StopLiveVideo(self._hcam, ueye.IS_FORCE_VIDEO_STOP)
        except Exception:
            pass
        if self._mem_ptr is not None:
            try:
                ueye.is_FreeImageMem(self._hcam, self._mem_ptr, self._mem_id)
            except Exception:
                pass
            self._mem_ptr = None
            self._mem_id  = None
        try:
            ueye.is_ExitCamera(self._hcam)
        except Exception as e:
            print(f"Warning: is_ExitCamera failed: {e}")
        self._hcam = None
        print("Camera released.")

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _apply_settings(self):
        if self.exposure_us is None:
            # exposure_us=None means: keep whatever exposure the sensor
            # currently has (e.g. dialed in during a visualizer session);
            # read it back so callers can report/save it.
            cur = ueye.double()
            ueye.is_Exposure(self._hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, cur, 8)
            self.exposure_us = float(cur.value) * 1000.0
            print(f"Using current sensor exposure: {self.exposure_us:.0f} µs")
        else:
            self.set_exposure(self.exposure_us)
        self.set_gain(self.gain_db)

    def set_exposure(self, exposure_us: float):
        self.exposure_us = exposure_us
        if self._hcam is None:
            return
        exposure_ms = ueye.double(exposure_us / 1000.0)
        ueye.is_Exposure(self._hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_ms, 8)

    def set_gain(self, gain_db: float):
        self.gain_db = gain_db
        if self._hcam is None:
            return
        # uEye gain is 0-100 integer; treat gain_db as a 0-100 master gain value
        gain_int = max(0, min(100, int(gain_db)))
        ueye.is_SetHardwareGain(self._hcam, gain_int,
                                ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER,
                                ueye.IS_IGNORE_PARAMETER)

    def set_roi(self, x: int, y: int, w: int, h: int):
        """Crop the sensor readout to a region (fast readout for small ROIs).

        Coordinates are snapped to the uEye AOI grid (x/width multiples of 8,
        y/height multiples of 2). The image memory is reallocated for the new
        size. Returns the actual (x, y, w, h) applied. A fresh connect()
        resets the camera to full frame (uEye re-inits to defaults).
        """
        if self._hcam is None:
            raise RuntimeError("Camera not connected. Call connect() first.")
        x, w = (x // 8) * 8, max((w // 8) * 8, 8)
        y, h = (y // 2) * 2, max((h // 2) * 2, 2)

        rect = ueye.IS_RECT()
        rect.s32X, rect.s32Y = ueye.int(x), ueye.int(y)
        rect.s32Width, rect.s32Height = ueye.int(w), ueye.int(h)
        ret = ueye.is_AOI(self._hcam, ueye.IS_AOI_IMAGE_SET_AOI,
                          rect, ueye.sizeof(rect))
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_AOI failed (code {ret}) for "
                               f"x={x} y={y} w={w} h={h}.")

        # Reallocate image memory for the new frame size.
        ueye.is_FreeImageMem(self._hcam, self._mem_ptr, self._mem_id)
        self._width, self._height = w, h
        self._mem_ptr = ueye.c_mem_p()
        self._mem_id = ueye.int()
        ueye.is_AllocImageMem(self._hcam, self._width, self._height,
                              self._bits_per_pixel, self._mem_ptr, self._mem_id)
        ueye.is_SetImageMem(self._hcam, self._mem_ptr, self._mem_id)
        print(f"ROI: x={x} y={y} {w}x{h}")
        return x, y, w, h

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Capture one frame and return it as an HxW monochrome numpy array (uint8 or uint16)."""
        if self._hcam is None:
            raise RuntimeError("Camera not connected. Call connect() first.")

        ret = ueye.is_FreezeVideo(self._hcam, ueye.IS_WAIT)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_FreezeVideo failed (code {ret}).")

        arr = np.zeros((self._height, self._width), dtype=self._dtype)
        ueye.is_CopyImageMem(self._hcam, self._mem_ptr, self._mem_id,
                             arr.ctypes.data_as(ctypes.POINTER(ctypes.c_char)))

        return arr.copy()

    def capture_to_buffer(self) -> np.ndarray:
        """Capture one frame into the internal buffer. Call save_stack() when done."""
        frame = self.capture()
        self._buffer.append(frame)
        print(f"  Buffered frame {len(self._buffer)}  shape={frame.shape}  dtype={frame.dtype}")
        return frame

    def save_stack(self, filename: str, save_previews: bool = False) -> np.ndarray:
        """Stack buffered frames into (H, W, N), save as .npy, clear buffer."""
        if not self._buffer:
            raise RuntimeError("Buffer is empty — nothing to save.")
        stack = np.stack(self._buffer, axis=-1)
        out = self.save_dir / filename
        if out.suffix != ".npy":
            out = out.with_suffix(".npy")
        np.save(out, stack)
        print(f"Saved stack: {out}  shape={stack.shape}  dtype={stack.dtype}")

        if save_previews:
            preview_dir = self.save_dir / "captured_images"
            preview_dir.mkdir(exist_ok=True)
            stem = out.stem
            for k in range(stack.shape[-1]):
                frame = stack[:, :, k] if stack.ndim == 3 else stack[:, :, :, k]
                if frame.ndim == 3:
                    Image.fromarray(frame).save(preview_dir / f"{stem}_frame{k:04d}.png")
                else:
                    mn, mx = frame.min(), frame.max()
                    frame8 = (
                        ((frame.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                        if mx > mn else frame.astype(np.uint8)
                    )
                    Image.fromarray(frame8).save(preview_dir / f"{stem}_frame{k:04d}.png")
            print(f"Previews saved: {preview_dir}  ({stack.shape[-1]} PNGs)")

        self._buffer.clear()
        return stack

    def clear_buffer(self):
        n = len(self._buffer)
        self._buffer.clear()
        print(f"Buffer cleared ({n} frames discarded).")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, image: np.ndarray, filename: str):
        out = self.save_dir / filename
        if out.suffix == ".npy":
            np.save(out, image)
            print(f"Saved .npy: {out}  shape={image.shape}  dtype={image.dtype}")
            return
        Image.fromarray(image.squeeze()).save(out)
        print(f"Saved: {out}")

    def timestamped_filename(self, prefix: str = "capture", ext: str = "png") -> str:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def scan():
        """Print all IDS uEye cameras visible to the driver."""
        if not _UEYE_AVAILABLE:
            print("pyueye not installed — cannot scan for IDS cameras.")
            return
        cam_list = ueye.UEYE_CAMERA_LIST()
        ueye.is_GetCameraList(cam_list)
        count = int(cam_list.dwCount)
        print(f"=== IDS uEye Cameras ({count} found) ===")
        for i in range(count):
            c = cam_list.uci[i]
            print(f"  ID={c.dwCameraID}  serial={c.SerNo.decode()}  "
                  f"model={c.Model.decode()}")
