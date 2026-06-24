from depth_from_focus import compute_mean_diff
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ── Settings ──────────────────────────────────────────────────────────────────
STACK_PATH = Path("coin_captures_ids_3250/stack.npy")

# ──────────────────────────────────────────────────────────────────────────────

#%%
frames = np.load(STACK_PATH)
print(f"Loaded: {STACK_PATH}  shape={frames.shape}")

_, md_image, _ = compute_mean_diff(frames[:, :, ::2], patch_size=5, avg_type="local", temporal_window=20)

depth  = np.argmax(md_image, axis=2)
maxamp = np.max(md_image, axis=2)

np.save(STACK_PATH.with_name(STACK_PATH.stem + "_depth.npy"),  depth)
np.save(STACK_PATH.with_name(STACK_PATH.stem + "_maxamp.npy"), maxamp)
print(f"Saved: {STACK_PATH.stem}_depth.npy  {STACK_PATH.stem}_maxamp.npy")
print(f"Depth range: {depth.min()} – {depth.max()} frames")

DEPTH_PATH = STACK_PATH.with_name(STACK_PATH.stem + "_depth.npy")
# %%
DEPTH_MIN  = None      # set e.g. 20, or None for auto
DEPTH_MAX  = None
CMAP       = "viridis"
# ──────────────────────────────────────────────────────────────────────────────

#%%
depth = np.load(DEPTH_PATH)
print(f"Loaded: {DEPTH_PATH}  shape={depth.shape}")
print(f"Data range: {depth.min()} – {depth.max()} frames")

vmin = DEPTH_MIN if DEPTH_MIN is not None else depth.min()
vmax = DEPTH_MAX if DEPTH_MAX is not None else depth.max()
print(f"Colormap clim: {vmin} – {vmax}")

#%%
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(depth, cmap=CMAP, vmin=vmin, vmax=vmax)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Depth (frame index)")
ax.set_title("Depth map")
ax.axis("off")
plt.tight_layout()

out = DEPTH_PATH.with_name(DEPTH_PATH.stem + "_render.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
