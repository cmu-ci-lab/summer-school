"""Re-render an existing OCT depth map with adjustable color limits.

Rendering lives in oct.save_colormap; this script only picks the file and the
vmin/vmax window.
"""
import numpy as np
from pathlib import Path
from oct import save_colormap

# ── Settings ──────────────────────────────────────────────────────────────────
DEPTH_PATH = Path("coin_captures_ids_new_test/stack_20260626_155621_depth.npy")

DEPTH_MIN  = 60      # set e.g. 20, or None for auto
DEPTH_MAX  = 80
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
save_colormap(depth, DEPTH_PATH.with_name(DEPTH_PATH.stem + "_render.png"),
              cmap=CMAP, vmin=vmin, vmax=vmax, title="Depth map",
              colorbar_label="Depth (frame index)", show=True)

# %%
