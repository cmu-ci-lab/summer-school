#%%
from pylablib.devices import Thorlabs

_devices = Thorlabs.list_kinesis_devices()
_brushed = [(sn, desc) for sn, desc in _devices if "Brushed Motor" in desc]
serial_number, device_desc = _brushed[0]
print(f"Connecting to: {device_desc}  (serial {serial_number})\n")

# Connect without scale so everything comes back in raw device units
stage = Thorlabs.KinesisMotor(serial_number)

#%%
info = stage.get_device_info()
print("=== Device Info ===")
print(f"  Model:          {info.model_number}")
print(f"  Serial:         {info.serial_number}")
print(f"  Firmware:       {info.firmware_version}")
print(f"  Hardware:       {info.hw_version}")
print(f"  Mod state:      {info.mod_state}")
print(f"  Channels:       {info.nchannel}")

#%%
print("\n=== Motor Parameters ===")
try:
    mp = stage.get_motor_params()
    print(f"  {mp}")
except Exception as e:
    print(f"  (not available: {e})")

print("\n=== Velocity Parameters (raw) ===")
try:
    vp = stage.get_velocity_params()
    print(f"  {vp}")
except Exception as e:
    print(f"  (not available: {e})")

print("\n=== Current position (raw encoder counts) ===")
print(f"  {stage.get_position()}")

print("\n=== Status ===")
print(f"  {stage.get_status()}")

stage.close()

# %%
