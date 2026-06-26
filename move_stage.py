#%%
from stage import ThorlabsStage
from camera import Camera

#%%
stage = ThorlabsStage(units="mm")
stage.connect()

cam = Camera(exposure_us=8000, gain_db=0.0, save_dir="coin_captures_ids_test")
cam.connect()

#%%
stage.home()

#%%
for i in range(200):
    stage.move_to(i * 0.005 + 18)   # steps of 5 µm starting at 18 mm
    cam.capture_to_buffer()
    print(f"  {stage.position:.3f} mm")

cam.save_stack(cam.timestamped_filename(prefix="stack", ext="npy"), save_previews=False)

#%%
stage.release()
cam.release()

# %%
