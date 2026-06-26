import sys
import time
from pylablib.devices import Thorlabs


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
        if sys.platform == "darwin":
            self._connect_macos()
        else:
            self._connect_usb()
        return self

    def _connect_usb(self):
        """Windows / Linux: USB APT direct via libusb."""
        devices = Thorlabs.list_kinesis_devices()
        brushed = [(sn, desc) for sn, desc in devices if "Brushed Motor" in desc]

        if not devices:
            raise RuntimeError("No Thorlabs Kinesis devices found. Check USB and APT driver.")
        if not brushed:
            found = ", ".join(f"{sn} ({d})" for sn, d in devices)
            raise RuntimeError(f"No Brushed Motor Controller found. Connected: {found}")
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

    def wait_move(self):
        self._stage.wait_move()


# ----------------------------------------------------------------------
# Quick test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    with ThorlabsStage(units="mm") as stage:
        stage.home()
        for i in range(5):
            stage.move_to(i * 2)
            print(f"  position: {stage.position:.3f} mm")
            time.sleep(0.3)
