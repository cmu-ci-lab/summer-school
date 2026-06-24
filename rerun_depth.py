import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────────
DEPTH_PATH = Path("coin_captures_0161H/stack_depth.npy")

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

# %%
