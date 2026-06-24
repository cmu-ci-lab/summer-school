#%%
import serial.tools.list_ports

ports = list(serial.tools.list_ports.comports())

print(f"Found {len(ports)} serial port(s):\n")
for p in ports:
    print(f"  Port:        {p.device}")
    print(f"  Description: {p.description}")
    print(f"  HWID:        {p.hwid}")
    print(f"  VID:PID:     {p.vid}:{p.pid}" if p.vid else "  VID:PID:     (not available)")
    print(f"  Manufacturer:{p.manufacturer}")
    print(f"  Serial No:   {p.serial_number}")
    print()

# Thorlabs APT devices use VID 0x0403 (FTDI chip)
THORLABS_VID = 0x0403
matches = [p for p in ports if p.vid == THORLABS_VID]

if matches:
    print("--- Likely Thorlabs device(s) (FTDI VID 0x0403) ---")
    for p in matches:
        print(f"  -> Use port: {p.device}  ({p.description})")
    print()
    print("Suggested pylablib line:")
    for p in matches:
        print(f'  stage = Thorlabs.KinesisMotor("{p.device}", scale=(409600, 21987328, 4506))')
else:
    print("No FTDI/Thorlabs device detected.")
    print("Make sure the USB cable is connected and the Thorlabs driver (APT/Kinesis) is installed.")

# %%
