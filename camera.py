"""Generic camera wrapper — auto-detects Allied Vision (vmbpy) or IDS (ids_peak).

Usage::
    from camera import Camera

    # Auto-detect (tries AVT first, then IDS):
    with Camera(exposure_us=10000, save_dir="captures") as cam:
        image = cam.capture()

    # Force a specific vendor:
    cam = Camera(exposure_us=10000, prefer="ids")

    # GigE AVT by IP, or IDS by serial:
    cam = Camera(exposure_us=10000, ip="192.168.1.100")
    cam = Camera(exposure_us=10000, serial="4103069584")

All methods (capture, capture_to_buffer, save_stack, set_exposure, …) are
forwarded transparently to the underlying AVTCamera or IDSCamera instance.
"""

from __future__ import annotations


_VENDORS = ("avt", "ids")


class Camera:
    """Auto-detecting camera wrapper.

    Parameters
    ----------
    exposure_us : float
        Exposure time in microseconds.
    gain_db : float
        Analogue gain in dB.
    save_dir : str
        Directory for saved images and stacks.
    ip : str | None
        If set, connect to a GigE AVT camera at this IP address.
    serial : str | None
        If set, connect to an IDS camera with this serial number.
    prefer : str | None
        ``"avt"`` or ``"ids"`` to skip auto-detection and go straight to
        that vendor.  Default ``None`` tries AVT first, then IDS.
    """

    def __init__(self, exposure_us: float = 10000, gain_db: float = 0.0,
                 save_dir: str = "captures",
                 ip: str | None = None,
                 serial: str | None = None,
                 prefer: str | None = None):
        if prefer is not None and prefer not in _VENDORS:
            raise ValueError(f"prefer must be one of {_VENDORS}, got {prefer!r}")
        self._kwargs = dict(exposure_us=exposure_us, gain_db=gain_db, save_dir=save_dir)
        self._ip     = ip
        self._serial = serial
        self._prefer = prefer
        self._cam    = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        order = [self._prefer] if self._prefer else list(_VENDORS)

        # If an ip is given, skip straight to AVT; serial → IDS
        if self._ip and "avt" not in order:
            order = ["avt"]
        if self._serial and "ids" not in order:
            order = ["ids"]

        errors: dict[str, str] = {}
        for vendor in order:
            try:
                cam = self._make(vendor)
                cam.connect()
                self._cam = cam
                return self
            except Exception as exc:
                errors[vendor] = str(exc)

        lines = "\n".join(f"  [{v}] {e}" for v, e in errors.items())
        raise RuntimeError(f"No supported camera found.\n{lines}")

    def _make(self, vendor: str):
        if vendor == "avt":
            from avt_camera import AVTCamera
            return AVTCamera(**self._kwargs, ip=self._ip)
        if vendor == "ids":
            from ids_camera import IDSCamera
            return IDSCamera(**self._kwargs, serial=self._serial)
        raise ValueError(f"Unknown vendor: {vendor!r}")

    def release(self):
        if self._cam is not None:
            self._cam.release()
            self._cam = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Transparent delegation — all other methods forwarded to _cam
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Only called when normal attribute lookup fails (i.e. _cam is set).
        # AttributeError (not RuntimeError) so getattr(cam, name, default) and
        # hasattr() probes on an unconnected wrapper fall back instead of raising.
        cam = object.__getattribute__(self, "_cam")
        if cam is None:
            raise AttributeError(
                f"Camera not connected. Call connect() before accessing {name!r}."
            )
        return getattr(cam, name)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def scan():
        """Scan for all supported cameras and print a summary."""
        print("=== Scanning for Allied Vision cameras (vmbpy) ===")
        try:
            from avt_camera import AVTCamera
            AVTCamera.scan()
        except ImportError:
            print("  vmbpy not installed — skipping AVT scan.")
        except Exception as exc:
            print(f"  AVT scan error: {exc}")

        print()
        print("=== Scanning for IDS cameras (ids_peak) ===")
        try:
            from ids_camera import IDSCamera
            IDSCamera.scan()
        except ImportError:
            print("  ids_peak not installed — skipping IDS scan.")
        except Exception as exc:
            print(f"  IDS scan error: {exc}")


# ----------------------------------------------------------------------
# Quick test when run directly:  python camera.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    Camera.scan()
