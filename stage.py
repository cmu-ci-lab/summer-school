import sys

# Guarded so this module stays importable (for TRAVEL_MM / resolve_target) on a
# machine with no stage drivers — visualizer.py runs camera-only in that case.
# Only connect() actually needs pylablib.
try:
    from pylablib.devices import Thorlabs
except ImportError:
    Thorlabs = None

# Usable travel of the actuator (Z825 / LTS150 short axis). Single source of
# truth — visualizer.py imports this rather than keeping its own copy.
TRAVEL_MM = 25.0


def resolve_target(text, current_mm, travel_mm=TRAVEL_MM):
    """Parse a typed position into an absolute target in mm.

    A leading '+' or '-' means a move RELATIVE to `current_mm` ("+0.2", "-1.5");
    anything else is absolute ("3.75"). The two can't be confused, because a
    negative absolute position is never valid.

    Returns (target_mm, is_relative). Raises ValueError with a message meant to
    be shown to the user if the text isn't a number, or the resulting position
    would leave the travel range.
    """
    text = text.strip()
    if not text:
        raise ValueError("no position entered")
    relative = text[0] in "+-"
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    target = current_mm + value if relative else value
    if not 0.0 <= target <= travel_mm:
        where = f"{current_mm:.4f} {value:+g} = {target:.4f}" if relative \
                else f"{target:.4f}"
        raise ValueError(f"{where} mm is outside the travel range "
                         f"(0 – {travel_mm:g} mm)")
    return target, relative


class ThorlabsStage:
    """Thorlabs brushed-motor stage wrapper (KDC101 + auto-detected stage).

    Units can be "m" or "mm". All position/velocity/acceleration arguments
    and return values use the chosen unit.

    Windows / Linux: connects via USB APT direct (pylablib + libusb).
    macOS: Apple's FTDI driver ignores Thorlabs' custom PID (0xfaf0), so no
    /dev/cu.usbserial-* appears. We drive the chip with pyftdi over libusb and
    hand pylablib an 'ftdi://' URL — no kext or driver install required.

    Usage::
        with ThorlabsStage(units="mm") as stage:
            stage.home()
            stage.move_to(10)        # 10 mm
            print(stage.position)    # mm

    Also runnable as a CLI (only when nothing else holds the stage — the
    controller allows a single connection, so quit visualizer.py first)::
        python stage.py --to 3.75    # absolute move, mm
        python stage.py --by -0.2    # relative move
        python stage.py --status     # position + homing state
    """

    _UNIT_SCALE = {"m": 1.0, "mm": 1e3}   # pylablib uses metres; multiply to get user unit
    _THORLABS_VID = 0x0403  # FTDI VID shared by all KDC101 / APT USB devices
    _THORLABS_PID = 0xfaf0  # Thorlabs' custom FTDI product ID (KDC101 / APT)

    def __init__(self, units: str = "mm"):
        if units not in self._UNIT_SCALE:
            raise ValueError(f"units must be 'm' or 'mm', got {units!r}")
        self.units = units
        self._scale = self._UNIT_SCALE[units]
        self._stage = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if Thorlabs is None:
            raise RuntimeError(
                "pylablib is not installed — cannot talk to the stage.\n"
                "Install the project environment (see setup.sh / environment.yml).")
        if sys.platform == "darwin":
            self._connect_macos()
        else:
            self._connect_usb()
        return self

    def _connect_usb(self):
        """Windows / Linux: USB APT direct via libusb."""
        devices = Thorlabs.list_kinesis_devices()

        def is_kdc101(sn, desc):
            # The description varies with firmware/driver era ("Brushed Motor
            # Controller", "APT DC Motor Controller", sometimes just
            # "KDC101"), so match loosely — and by serial prefix: KDC101
            # serial numbers always start with 27.
            d = (desc or "").lower()
            return ("brushed motor" in d or "dc motor" in d
                    or "kdc101" in d or str(sn).startswith("27"))

        brushed = [(sn, desc) for sn, desc in devices if is_kdc101(sn, desc)]

        if not devices:
            raise RuntimeError("No Thorlabs Kinesis devices found. Check USB and APT driver.")
        if not brushed:
            found = ", ".join(f"{sn} ({d})" for sn, d in devices)
            raise RuntimeError(
                "No KDC101 (brushed motor controller) recognized among the "
                f"Kinesis devices. Connected: {found}\n"
                "If one of these IS the KDC101, its description/serial is "
                "unexpected — please report the line above.")
        if len(brushed) > 1:
            print(f"Warning: {len(brushed)} brushed controllers found, using first.")

        sn, desc = brushed[0]
        self._stage = Thorlabs.KinesisMotor(sn, scale="stage")

        info = self._stage.get_device_info()
        print(f"Connected: {desc}  model={info.model_no}  serial={info.serial_no}")
        print(f"Stage: {self._stage.get_stage()}  |  internal units: {self._stage.get_scale_units()}  |  user units: {self.units}")

    def _connect_macos(self):
        """macOS: drive the FTDI chip with pyftdi (libusb) — NOT a serial port.

        The KDC101 uses Thorlabs' custom FTDI PID (0xfaf0). On macOS:
          * Apple's FTDI driver only binds standard PIDs, so NO /dev/cu.usbserial-*
            is ever created — the serial-port approach can't find the device.
          * pylablib's default ft232 backend (pyft232 + old libftdi ABI) is broken
            on arm64 / libftdi 1.5.
        So we use pyftdi (pure-Python, libusb-based, supports custom PIDs). pyftdi
        registers an 'ftdi://' URL handler with pyserial; passing ("serial", url)
        forces pylablib's serial backend over pyftdi and reuses all its motor logic.

        Requires (already in environment.yml): conda-forge libusb + pip pyftdi.
        """
        try:
            from pyftdi.ftdi import Ftdi
            import pyftdi.serialext  # noqa: F401 — import registers 'ftdi://' with pyserial
        except ImportError as e:
            raise RuntimeError(
                "pyftdi is required on macOS but is not installed.\n"
                "Install with: pip install pyftdi   (and conda install -c conda-forge libusb)"
            ) from e

        # Register Thorlabs' custom PID so pyftdi will recognise the controller.
        # add_custom_product raises if already registered — harmless on repeat calls.
        try:
            Ftdi.add_custom_product(vid=self._THORLABS_VID, pid=self._THORLABS_PID)
        except ValueError:
            pass

        # Verify the device is actually on the bus, with a clear error if not.
        devices = Ftdi.list_devices()
        if not devices:
            raise RuntimeError(
                "No Thorlabs FTDI controller found on the USB bus.\n"
                f"Expected an FTDI device with VID 0x{self._THORLABS_VID:04x} "
                f"PID 0x{self._THORLABS_PID:04x} (KDC101).\n"
                "Check: cable plugged in, controller powered on.\n"
                "Diagnose: system_profiler SPUSBDataType | grep -i -A3 thorlabs"
            )
        if len(devices) > 1:
            print(f"Warning: {len(devices)} Thorlabs FTDI devices found, using first.")

        url = f"ftdi://0x{self._THORLABS_VID:x}:0x{self._THORLABS_PID:x}/1"
        print(f"macOS: connecting to KDC101 via {url}")
        # ("serial", url) forces pylablib's serial backend (over pyftdi), bypassing
        # the broken ft232 backend. scale="stage" auto-detects the actuator.
        self._stage = Thorlabs.KinesisMotor(("serial", url), scale="stage")

        info = self._stage.get_device_info()
        print(f"Connected: model={info.model_no}  serial={info.serial_no}  fw={info.fw_ver}")
        print(f"Stage: {self._stage.get_stage()}  |  internal units: {self._stage.get_scale_units()}  |  user units: {self.units}")

    def release(self):
        if self._stage is not None:
            self._stage.close()
            self._stage = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.release()

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def home(self, wait: bool = True):
        """Home the stage. Blocks until complete if wait=True."""
        self._stage.home(sync=wait)
        print(f"Homed. Position: {self.position:.4f} {self.units}")

    def home_if_needed(self, wait: bool = True) -> bool:
        """Home only if the controller hasn't been homed yet. Returns True if it did.

        Absolute positions are meaningless until the stage has a datum, but
        re-homing an already-homed stage just wastes a cycle and drives it back
        to zero — so callers that only care about *being* homed use this.
        """
        if self.is_homed:
            return False
        print("Stage is not homed (absolute positions need a datum) — homing...")
        self.home(wait=wait)
        return True

    def move_to(self, position, wait: bool = True):
        """Move to absolute position in user units. Blocks if wait=True."""
        self._stage.move_to(position / self._scale)
        if wait:
            self._stage.wait_move()

    def move_by(self, distance, wait: bool = True):
        """Move by a relative distance in user units. Blocks if wait=True."""
        self._stage.move_by(distance / self._scale)
        if wait:
            self._stage.wait_move()

    def stop(self):
        self._stage.stop()

    def get_velocity(self):
        """Current max move velocity in user units per second."""
        return self._stage.get_velocity_parameters().max_velocity * self._scale

    def set_velocity(self, max_velocity, acceleration=None):
        """Set the move velocity (user units/s; acceleration in units/s^2).

        Used e.g. by oct_crop_scan's continuous mode, where the frame spacing
        is velocity / camera fps. Persists until changed or power cycle —
        callers should restore the previous value (see get_velocity).
        """
        kwargs = {"max_velocity": max_velocity / self._scale}
        if acceleration is not None:
            kwargs["acceleration"] = acceleration / self._scale
        self._stage.setup_velocity(**kwargs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position(self):
        """Current position in user units."""
        return self._stage.get_position() * self._scale

    @property
    def is_moving(self) -> bool:
        return self._stage.is_moving()

    @property
    def is_homed(self) -> bool:
        """True if the controller has been homed since it was powered on.

        Reads the APT 'homed' status bit, which the controller clears on power
        cycle — so this answers "does the current position mean anything?".
        """
        return self._stage.is_homed()

    def wait_move(self):
        self._stage.wait_move()


# ----------------------------------------------------------------------
# CLI — move the stage from a terminal
# ----------------------------------------------------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Move the Thorlabs stage to a position, or report its state.",
        epilog="Note: the controller can only be opened by ONE process, so this "
               "cannot run while visualizer.py is holding the stage.")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--to", type=float, metavar="MM",
                      help=f"move to an absolute position in mm (0 – {TRAVEL_MM:g})")
    what.add_argument("--by", type=float, metavar="MM",
                      help="move by a relative distance in mm (may be negative)")
    what.add_argument("--status", action="store_true",
                      help="print the position and homing state; move nothing")
    homing = p.add_mutually_exclusive_group()
    homing.add_argument("--home", action="store_true",
                        help="home first, even if the stage is already homed")
    homing.add_argument("--no-home", action="store_true",
                        help="never home, even if the stage has not been homed "
                             "(the position will not be a true absolute)")
    args = p.parse_args()

    stage = ThorlabsStage(units="mm")
    try:
        stage.connect()
    except Exception as e:
        print(f"Could not connect to the stage: {e}")
        if Thorlabs is not None:
            # Drivers are present, so the likeliest cause is another process
            # holding the device. The underlying FTDI open failure is opaque —
            # say what it usually means.
            print("If visualizer.py (or another script) is running, quit it "
                  "first — the controller allows only one connection at a time.")
        return 1

    try:
        print(f"Position: {stage.position:.4f} mm    "
              f"homed: {'yes' if stage.is_homed else 'NO'}")
        if args.status:
            return 0

        if args.home:
            stage.home()
        elif not args.no_home:
            stage.home_if_needed()
        elif not stage.is_homed:
            print("Warning: stage is not homed and --no-home was given; the "
                  "position below is relative to wherever it powered on.")

        # Resolve both forms through the same parser the visualizer's typed
        # entry uses, so the travel-range check can't disagree between them.
        text = f"{args.by:+g}" if args.by is not None else f"{args.to:g}"
        try:
            target, relative = resolve_target(text, stage.position)
        except ValueError as e:
            print(f"Refusing to move: {e}")
            return 1

        kind = "by" if relative else "to"
        print(f"Moving {kind} {text} mm → {target:.4f} mm ...")
        stage.move_to(target)          # blocking: a CLI must not return early
        print(f"Position: {stage.position:.4f} mm")
        return 0
    finally:
        stage.release()


if __name__ == "__main__":
    sys.exit(main())
