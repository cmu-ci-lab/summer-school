#%%
from stage import ThorlabsStage
from camera import Camera

#%%
stage = ThorlabsStage(units="mm")
stage.connect()

cam = Camera(exposure_us=8000, gain_db=0.0, save_dir="coin_captures_ids_new_test")
cam.connect()

#%%
stage.home()
start_position = 1.50
steps = 0.001 
#%%
positions = []
for i in range(400):
    stage.move_to(i * steps + start_position)
    cam.capture_to_buffer()
    positions.append(stage.position)
    print(f"  {stage.position:.3f} mm")

stack_name = cam.timestamped_filename(prefix="stack", ext="npy")
cam.save_stack(stack_name, save_previews=False)

# Sidecar metadata so downstream tools (e.g. depth_to_pointcloud.py) know the
# scan geometry without re-connecting the hardware.
import json
from pathlib import Path
meta = {
    "step_mm": steps,
    "start_position_mm": start_position,
    "n_frames": len(positions),
    "positions_mm": positions,
    "exposure_us": cam.exposure_us,
    "pixel_size_um": getattr(cam, "pixel_size_um", None),
}
meta_path = Path(cam.save_dir) / (Path(stack_name).stem + "_meta.json")
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Saved metadata: {meta_path}")

#%%
stage.release()
cam.release()

# %%
