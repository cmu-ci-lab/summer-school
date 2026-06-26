import vmbpy
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime


class AVTCamera:
    """Allied Vision camera wrapper — USB and GigE (Ethernet) cameras.

    USB (auto-discover)::
        with AVTCamera(exposure_us=5000) as cam: ...

    GigE (connect by IP)::
        with AVTCamera(exposure_us=5000, ip="192.168.1.100") as cam: ...

    When ip=None the first Allied Vision camera found (USB or GigE) is used.
    GigE cameras get jumbo-frame packet-size applied automatically.
    """

    def __init__(self, exposure_us: float = 10000, gain_db: float = 0.0,
                 save_dir: str = "captures", ip: str | None = None):
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.save_dir = Path(save_dir)
        self.ip = ip          # GigE: "192.168.x.x"; None = auto-discover
        self._vmb = None
        self._cam = None
        self._buffer: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        self._vmb = vmbpy.VmbSystem.get_instance().__enter__()

        try:
            raw = self._find_by_ip(self.ip) if self.ip else self._find_by_vendor()
            self._cam = raw.__enter__()
        except Exception:
            # VmbSystem was started — shut it down so its threads don't block the process
            try:
                self._vmb.__exit__(None, None, None)
            except Exception:
                pass
            self._vmb = None
            raise

        is_gige = self._is_gige()
        transport = "GigE" if is_gige else "USB"
        print(f"Connected [{transport}]: {self._cam.DeviceModelName.get()}  (id: {self._cam.get_id()})")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {self.save_dir.resolve()}")
        self._apply_settings()
        self._apply_gige_settings()   # safe on USB — all wrapped in try/except
        return self

    def _find_by_ip(self, ip: str):
        """Locate a GigE camera by IP address."""
        # Vimba X accepts the IP string directly as a camera ID for GigE TL
        try:
            return self._vmb.get_camera_by_id(ip)
        except Exception:
            pass
        # Fall back: scan all cameras and compare GevDeviceIPAddress
        for c in self._vmb.get_all_cameras():
            with c:
                try:
                    if self._int_to_ip(c.GevDeviceIPAddress.get()) == ip:
                        return c
                except Exception:
                    pass
        raise RuntimeError(
            f"No GigE camera found at {ip}.\n"
            "Check: cable plugged in, camera powered, Vimba X GigE Transport Layer installed,\n"
            "       NIC IP in same subnet (e.g. 192.168.1.x for camera at 192.168.1.100)."
        )

    def _find_by_vendor(self):
        """Auto-discover the first Allied Vision camera (USB or GigE)."""
        cameras = self._vmb.get_all_cameras()
        if not cameras:
            raise RuntimeError(
                "No cameras found. Check USB/Ethernet connection and Vimba X Transport Layers."
            )
        avt = []
        for c in cameras:
            with c:
                if "Allied Vision" in c.DeviceVendorName.get():
                    avt.append(c)
        if not avt:
            ids = [c.get_id() for c in cameras]
            raise RuntimeError(f"No Allied Vision camera found. Detected IDs: {', '.join(ids)}")
        if len(avt) > 1:
            print(f"Warning: {len(avt)} Allied Vision cameras found, using first.")
        return avt[0]

    def _is_gige(self) -> bool:
        # GevSCPSPacketSize is a GigE-only stream feature — more reliable indicator
        # than GevDeviceIPAddress which may not be directly accessible on all cameras.
        return getattr(self._cam, "GevSCPSPacketSize", None) is not None

    @staticmethod
    def _int_to_ip(n: int) -> str:
        return ".".join(str((n >> (8 * i)) & 0xFF) for i in (3, 2, 1, 0))

    @staticmethod
    def _same_subnet(ip_a: str, ip_b: str, mask: str) -> bool:
        import ipaddress
        try:
            net = ipaddress.IPv4Network(f"{ip_a}/{mask}", strict=False)
            return ipaddress.IPv4Address(ip_b) in net
        except Exception:
            return False

    @staticmethod
    def scan():
        """Print all cameras and GigE NIC state visible to Vimba X.

        Run this before connect() to diagnose GigE subnet mismatches.
        """
        with vmbpy.VmbSystem.get_instance() as vmb:
            # ── GigE transport interfaces (NICs) ──────────────────────────
            print("=== GigE Transport Interfaces (NICs) ===")
            gige_nics: list[tuple[str, str, str]] = []   # (name, ip, mask)
            for iface in vmb.get_all_interfaces():
                with iface:
                    name = iface.get_id()
                    try:
                        ip   = AVTCamera._int_to_ip(iface.GevInterfaceSubnetIPAddress.get())
                        mask = AVTCamera._int_to_ip(iface.GevInterfaceSubnetMask.get())
                        print(f"  {name:30s}  {ip:16s}  mask {mask}")
                        gige_nics.append((name, ip, mask))
                    except Exception:
                        pass   # non-GigE interface, skip

            if not gige_nics:
                print("  (none — Vimba X GigE Transport Layer may not be installed)")

            # ── Cameras ───────────────────────────────────────────────────
            print("\n=== Cameras ===")
            cameras = vmb.get_all_cameras()
            if not cameras:
                print("  No cameras found.")
                return

            for c in cameras:
                with c:
                    try:
                        vendor = c.DeviceVendorName.get()
                        model  = c.DeviceModelName.get()
                    except Exception:
                        vendor, model = "?", "?"

                    try:
                        cam_ip = AVTCamera._int_to_ip(c.GevDeviceIPAddress.get())
                        # Check subnet match against known NICs
                        matched = [
                            f"{nic_ip} ({name})"
                            for name, nic_ip, mask in gige_nics
                            if AVTCamera._same_subnet(nic_ip, cam_ip, mask)
                        ]
                        subnet_note = (
                            f"  ✓ reachable via {matched[0]}" if matched
                            else "  ✗ NO NIC in same subnet — set NIC IP to same /24 as camera"
                        )
                        print(f"  [GigE] {vendor} {model}")
                        print(f"         camera IP : {cam_ip}")
                        print(f"         {subnet_note}")
                        print(f"         use: AVTCamera(ip={cam_ip!r})")
                    except Exception:
                        print(f"  [USB]  {vendor} {model}  id={c.get_id()}")

    def _apply_gige_settings(self):
        """Conservative GigE stream settings to prevent packet loss / banding."""
        # Packet size: stay at standard MTU — jumbo frames need switch + NIC config.
        applied = []
        try:
            self._cam.GevSCPSPacketSize.set(1500)
            actual = self._cam.GevSCPSPacketSize.get()
            applied.append(f"PacketSize={actual}")
        except Exception:
            pass

        try:
            self._cam.GevSCPD.set(10000)
            actual = self._cam.GevSCPD.get()
            applied.append(f"InterPacketDelay={actual}ns")
        except Exception:
            pass

        if applied:
            print(f"GigE stream settings applied: {', '.join(applied)}")
        else:
            print("GigE stream settings: none applied (camera may not expose these features)")

    def release(self):
        if self._cam is not None:
            self._cam.__exit__(None, None, None)
            self._cam = None
        if self._vmb is not None:
            self._vmb.__exit__(None, None, None)
            self._vmb = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _set_feature(self, *names, value):
        """Try feature names in order; raises if none exist on this camera."""
        for name in names:
            feat = getattr(self._cam, name, None)
            if feat is not None:
                feat.set(value)
                return name
        raise AttributeError(
            f"Camera has none of {names!r}. "
            f"Run: [print(f.get_name()) for f in cam._cam.get_all_features()] to inspect."
        )

    def _set_best_mono_pixel_format(self):
        """Set the highest monochrome pixel format the camera natively supports."""
        preferred_enums = [
            vmbpy.PixelFormat.Mono16,
            vmbpy.PixelFormat.Mono12,
            vmbpy.PixelFormat.Mono10,
            vmbpy.PixelFormat.Mono8,
        ]
        
        # Try direct attribute access first
        feat = getattr(self._cam, "PixelFormat", None)
        if feat is not None:
            print("Attempting to set PixelFormat to highest supported mono format...")
            for pref_enum in preferred_enums:
                try:
                    feat.set(pref_enum)
                    print(f"✓ Set PixelFormat to: {pref_enum}")
                    return
                except Exception as e:
                    print(f"  ✗ Could not set to {pref_enum}: {e}")
                    continue
            print("⚠ Could not set PixelFormat to any mono format — camera will use default")
            return
        
        # No direct PixelFormat attribute — scan all features
        print("PixelFormat not accessible as attribute. Scanning all camera features...")
        try:
            all_feats = self._cam.get_all_features()
            # Find features related to format/pixel/image
            format_feats = [f for f in all_feats if any(x in f.get_name() for x in ["Format", "Pixel", "Image", "Bayer"])]
            
            if format_feats:
                print(f"Found {len(format_feats)} format-related features:")
                for f in format_feats:
                    fname = f.get_name()
                    try:
                        val = f.get()
                        ftype = type(f).__name__
                        print(f"  • {fname} ({ftype}): {val}")
                    except Exception as e:
                        print(f"  • {fname}: <error reading: {e}>")
                
                # Try to set any feature named *Format* to a Mono value
                for f in format_feats:
                    if "Format" in f.get_name():
                        for pref_enum in preferred_enums:
                            try:
                                f.set(pref_enum)
                                print(f"✓ Set {f.get_name()} to {pref_enum}")
                                return
                            except Exception:
                                continue
            else:
                print("No format-related features found on this camera.")
        except Exception as e:
            print(f"Error scanning features: {e}")
        
        print("⚠ Cannot configure pixel format — camera will use its default")

    def _apply_settings(self):
        self._set_best_mono_pixel_format()

        # Auto-off: try both SFNC 2.x and older SFNC 1.x names
        try:
            self._set_feature("ExposureAuto", value="Off")
        except AttributeError:
            pass   # some cameras have no auto-exposure toggle

        name = self._set_feature("ExposureTime", "ExposureTimeAbs", value=self.exposure_us)
        self._exposure_feature = name

        try:
            self._set_feature("GainAuto", value="Off")
        except AttributeError:
            pass

        try:
            name = self._set_feature("Gain", "GainRaw", value=self.gain_db)
            self._gain_feature = name
        except AttributeError:
            self._gain_feature = None
            print("Warning: no Gain feature found on this camera — gain not set.")

    def set_exposure(self, exposure_us: float):
        self.exposure_us = exposure_us
        if self._cam:
            self._set_feature("ExposureTime", "ExposureTimeAbs", value=exposure_us)

    def set_gain(self, gain_db: float):
        self.gain_db = gain_db
        if self._cam:
            try:
                self._set_feature("Gain", "GainRaw", value=gain_db)
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    FRAME_TIMEOUT_MS = 5000   # per-frame timeout (ms)

    def capture(self) -> np.ndarray:
        """Return raw frame as a numpy array (uint8 or uint16, HxW or HxWx3)."""
        if self._cam is None:
            raise RuntimeError("Camera not connected. Call connect() first.")

        frame = self._cam.get_frame(timeout_ms=self.FRAME_TIMEOUT_MS)

        fmt = frame.get_pixel_format()
        mono_fmts  = (vmbpy.PixelFormat.Mono8,  vmbpy.PixelFormat.Mono10,
                      vmbpy.PixelFormat.Mono12, vmbpy.PixelFormat.Mono16)
        color_fmts = (vmbpy.PixelFormat.Rgb8,)

        # Log format on first capture for diagnostic purposes
        if not hasattr(self, '_format_logged'):
            print(f"[Camera] Receiving frames in format: {fmt}")
            self._format_logged = True

        if fmt not in mono_fmts and fmt not in color_fmts:
            target = vmbpy.PixelFormat.Rgb8 if "Bayer" in str(fmt) else vmbpy.PixelFormat.Mono8
            frame = frame.convert_pixel_format(target)

        return frame.as_numpy_ndarray().squeeze()

    def capture_8bit(self) -> np.ndarray:
        """Return frame rescaled to uint8 (full dynamic range mapped to 0–255)."""
        arr = self.capture()
        if arr.dtype == np.uint8:
            return arr
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr.astype(np.float32) - mn) / (mx - mn) * 255
        return arr.astype(np.uint8)

    def capture_to_buffer(self) -> np.ndarray:
        """Capture one frame, append it to the internal buffer, and return it.

        Call save_stack() when done to flush the buffer to a .npy file.
        """
        frame = self.capture()
        self._buffer.append(frame)
        print(f"  Buffered frame {len(self._buffer)}  shape={frame.shape}  dtype={frame.dtype}")
        return frame

    def save_stack(self, filename: str, save_previews: bool = False) -> np.ndarray:
        """Stack buffered frames into (H, W, N), save as .npy, and clear buffer.

        Args:
            filename:      output filename (forces .npy extension).
            save_previews: if True, also saves each frame as a rescaled 8-bit PNG
                           in save_dir/captured_images/.
        """
        if not self._buffer:
            raise RuntimeError("Buffer is empty — nothing to save.")
        stack = np.stack(self._buffer, axis=-1)   # (H, W, N)
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
                frame = stack[:, :, k]
                mn, mx = frame.min(), frame.max()
                frame8 = ((frame.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else frame.astype(np.uint8)
                Image.fromarray(frame8).save(preview_dir / f"{stem}_frame{k:04d}.png")
            print(f"Previews saved: {preview_dir}  ({stack.shape[-1]} PNGs)")

        self._buffer.clear()
        return stack

    def clear_buffer(self):
        """Discard buffered frames without saving."""
        n = len(self._buffer)
        self._buffer.clear()
        print(f"Buffer cleared ({n} frames discarded).")

    def capture_stack(self, n_frames: int, delay: float = 0.0) -> np.ndarray:
        """Capture n_frames and return a (H, W, N) array of raw values.

        Args:
            n_frames: number of frames to capture.
            delay:    seconds to wait between frames (0 = as fast as possible).
        """
        import time
        frames = []
        for i in range(n_frames):
            frames.append(self.capture())
            if delay > 0 and i < n_frames - 1:
                time.sleep(delay)
            print(f"\r  Captured {i+1}/{n_frames}", end="", flush=True)
        print()
        stack = np.stack(frames, axis=-1)   # (H, W, N)  — matches MATLAB frames(:,:,k)
        print(f"Stack shape: {stack.shape}  dtype: {stack.dtype}")
        return stack

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, image: np.ndarray, filename: str):
        """Save into save_dir.

        - ``.npy``  → raw numpy array (stack or single frame, any dtype)
        - 16-bit image → TIFF (full precision) + rescaled 8-bit PNG
        - 8-bit image  → PNG
        """
        out = self.save_dir / filename

        if out.suffix == ".npy":
            np.save(out, image)
            print(f"Saved .npy: {out}  shape={image.shape}  dtype={image.dtype}")
            return

        arr = image.squeeze()
        stem = (self.save_dir / out.stem).as_posix()

        if arr.dtype == np.uint16:
            tiff_path = stem + ".tiff"
            Image.fromarray(arr, mode="I;16").save(tiff_path)
            print(f"16-bit TIFF: {tiff_path}")

            mn, mx = arr.min(), arr.max()
            arr8 = ((arr.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else arr.astype(np.uint8)
            png_path = stem + "_8bit.png"
            Image.fromarray(arr8).save(png_path)
            print(f"8-bit  PNG:  {png_path}  (rescaled {mn}–{mx} → 0–255)")
        else:
            Image.fromarray(arr).save(out)
            print(f"Saved: {out}")

    def timestamped_filename(self, prefix: str = "capture", ext: str = "png") -> str:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


# ----------------------------------------------------------------------
# Quick test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    ip_arg = sys.argv[1] if len(sys.argv) > 1 else None  # optional: python capture_camera.py 192.168.1.100
    with AVTCamera(exposure_us=10000, gain_db=0.0, save_dir="captures", ip=ip_arg) as cam:
        image = cam.capture()
        print(f"Shape: {image.shape}  dtype: {image.dtype}")
        cam.save(image, cam.timestamped_filename())
