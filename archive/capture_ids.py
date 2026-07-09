from pyueye import ueye
import numpy as np
import ctypes
from PIL import Image
from datetime import datetime

hcam = ueye.HIDS(0)  # 0 = first available camera

ret = ueye.is_InitCamera(hcam, None)
if ret != ueye.IS_SUCCESS:
    raise RuntimeError(f"is_InitCamera failed: {ret}")

# Get sensor info
sensor_info = ueye.SENSORINFO()
ueye.is_GetSensorInfo(hcam, sensor_info)
print(f"Connected: {sensor_info.strSensorName.decode()}  "
      f"{sensor_info.nMaxWidth}x{sensor_info.nMaxHeight}")

# Set color mode to BGR8
ueye.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)

# Set exposure (ms)
exposure_ms = ueye.double(15.0)
ueye.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, exposure_ms, 8)

# Allocate image memory
width  = int(sensor_info.nMaxWidth)
height = int(sensor_info.nMaxHeight)
bitspixel = 24  # BGR8
mem_ptr  = ueye.c_mem_p()
mem_id   = ueye.int()
ueye.is_AllocImageMem(hcam, width, height, bitspixel, mem_ptr, mem_id)
ueye.is_SetImageMem(hcam, mem_ptr, mem_id)

# Capture a single frame
ret = ueye.is_FreezeVideo(hcam, ueye.IS_WAIT)
if ret != ueye.IS_SUCCESS:
    raise RuntimeError(f"is_FreezeVideo failed: {ret}")

# Copy to numpy array
arr = np.zeros((height, width, 3), dtype=np.uint8)
ueye.is_CopyImageMem(hcam, mem_ptr, mem_id, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_char)))

# Convert BGR → RGB and save
img = Image.fromarray(arr[:, :, ::-1])
filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
img.save(filename)
print(f"Saved: {filename}  shape={arr.shape}")

# Clean up
ueye.is_FreeImageMem(hcam, mem_ptr, mem_id)
ueye.is_ExitCamera(hcam)
