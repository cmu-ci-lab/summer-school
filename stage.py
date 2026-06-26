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

    def __init__(self, units: str = "mm", stage: str = "stage"):
        """
        units : "m" or "mm" — units for all position/velocity arguments and returns.
        stage : how pylablib determines the motor's step scale.
                "stage"  -> auto-detect from the stage ID programmed into the
                            controller (set via Thorlabs Kinesis). This is a query
                            over the open connection, so it works regardless of how
                            we connected (serial port or APT serial number).
                Otherwise pass an explicit stage name when auto-detect fails, e.g.
                "Z825", "Z812", "Z806", "MTS25-Z8", "MTS50-Z8", "PRM1-Z8", "CR1-Z7".
        """
        if units not in self._UNIT_SCALE:
            raise ValueError(f"units must be 'm' or 'mm', got {units!r}")
        self.units = units
        self._scale = self._UNIT_SCALE[units]
        self._scale_spec = stage   # passed to KinesisMotor(scale=...)
        self._stage = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if sys.platform == "win32":
            self._connect_apt()       # Windows: APT enumeration by serial number
        else:
            self._connect_serial()    # Linux / macOS: open the FTDI serial port directly
        return self

    def _connect_serial(self):
        """Linux / macOS: connect through the FTDI serial port (e.g. /dev/ttyUSB0).

        The KDC101 enumerates as an FTDI USB-serial device (VID 0x0403). We find
        it by VID and open the port directly rather than using APT serial-number
        enumeration: that needs the ft232/libftdi backend, which isn't always
        present (Linux without libftdi; macOS where Apple's FTDI driver holds the
        device). The motor's step scale is still auto-detected by querying the
        controller over this connection.
        """
        import serial.tools.list_ports
        ftdi_ports = [p for p in serial.tools.list_ports.comports()
                      if p.vid == self._THORLABS_VID]

        if not ftdi_ports:
            raise RuntimeError(
                "No Thorlabs FTDI serial port found (VID 0x0403). Check the USB cable.\n"
                "Diagnose: python -c \""
                "import serial.tools.list_ports; "
                "[print(p.device, hex(p.vid or 0), p.description) "
                "for p in serial.tools.list_ports.comports()]\""
            )
        if len(ftdi_ports) > 1:
            print(f"Warning: {len(ftdi_ports)} FTDI ports found, using first: {ftdi_ports[0].device}")

        port = ftdi_ports[0].device
        print(f"Connecting via serial port {port}")
        self._open(port)

    def _connect_apt(self):
        """Windows: USB APT direct enumeration by serial number."""
        devices = Thorlabs.list_kinesis_devices()
        if not devices:
            raise RuntimeError("No Thorlabs Kinesis devices found. Check USB and APT driver.")

        brushed = [(sn, desc) for sn, desc in devices if "Brushed Motor" in desc]
        if not brushed:
            found = ", ".join(f"{sn} ({d})" for sn, d in devices)
            raise RuntimeError(f"No Brushed Motor Controller found. Connected: {found}")
        if len(brushed) > 1:
            print(f"Warning: {len(brushed)} brushed controllers found, using first.")

        sn, desc = brushed[0]
        print(f"Connecting to {desc} (serial {sn})")
        self._open(sn)

    def _open(self, conn):
        """Open the KinesisMotor on a connection (serial port or APT serial number)."""
        self._stage = Thorlabs.KinesisMotor(conn, scale=self._scale_spec)

        info = self._stage.get_device_info()
        detected = self._stage.get_stage()
        print(f"Connected: model={info.model_no}  serial={info.serial_no}")
        print(f"Stage: {detected}  |  internal units: {self._stage.get_scale_units()}  |  user units: {self.units}")

        if detected is None and self._scale_spec == "stage":
            print("WARNING: stage type not auto-detected — positions are in RAW STEPS, not mm.")
            print("         The controller has no stage programmed. Either set it once in")
            print("         Thorlabs Kinesis, or pass it explicitly, e.g.:")
            print("           ThorlabsStage(units='mm', stage='Z825')")
            print("         Common KDC101 stages: Z806 Z812 Z825 MTS25-Z8 MTS50-Z8 PRM1-Z8 CR1-Z7")

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
