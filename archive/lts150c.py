#%%
from pylablib.devices import Thorlabs
import time

stage = Thorlabs.KinesisMotor("/dev/ttyUSB0", scale = (409600, 21987328, 4506))
print("Connected:", stage.get_device_info())

stage.home(sync = True)
print("Homed. Position:", stage.get_position())

for i in range(1):
    stage.move_to(10)
    stage.wait_move()
    print("Now at:", stage.get_position())
    time.sleep(0.5)
stage.close()

# %%
