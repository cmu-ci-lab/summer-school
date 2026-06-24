import sys
import time
from pylablib.devices import Thorlabs


class ThorlabsStage:
    """Thorlabs brushed-motor stage wrapper (KDC101 + auto-detected stage).

    Units can be "m" or "mm". All position/velocity/acceleration arguments
    and return values use the chosen unit.

    Windows / Linux: connects via USB APT direct (pylablib + libusb).
    macOS: Apple's FTDI kernel driver claims the device and exposes it as
    /dev/cu.usbserial-*; we connect through that serial port instead —
    no kext unloading required.

    Usage::
        with ThorlabsStage(units="mm") as stage:
            stage.home()
            stage.move_to(10)        # 10 mm
            print(stage.position)    # mm
    """

    _UNIT_SCALE = {"m": 1.0, "mm": 1e3}   # pylablib uses metres; multiply to get user unit
    _THORLABS_VID = 0x0403  # FTDI VID shared by all KDC101 / APT USB devices

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
        """macOS: Apple's FTDI driver exposes /dev/cu.usbserial-* — use that directly.

        list_kinesis_devices() requires libusb and won't work while Apple's driver
        holds the device, so we scan serial ports by VID instead.
        """
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        ftdi_ports = [p for p in ports if p.vid == self._THORLABS_VID]

        if not ftdi_ports:
            raise RuntimeError(
                "No FTDI serial port found. Make sure the KDC101 is plugged in.\n"
                "Expected: /dev/cu.usbserial-* with VID 0x0403.\n"
                "Diagnose: python -c \""
                "import serial.tools.list_ports; "
                "[print(p.device, hex(p.vid or 0), p.description) "
                "for p in serial.tools.list_ports.comports()]\""
            )
        if len(ftdi_ports) > 1:
            print(f"Warning: {len(ftdi_ports)} FTDI ports found, using first: {ftdi_ports[0].device}")

        port = ftdi_ports[0].device
        print(f"macOS: connecting via serial port {port}")
        self._stage = Thorlabs.KinesisMotor(port, scale="stage")

        info = self._stage.get_device_info()
        print(f"Connected: model={info.model_no}  serial={info.serial_no}")
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
