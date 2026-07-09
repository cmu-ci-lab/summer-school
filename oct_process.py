from oct import compute_mean_diff
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ── Settings (defaults; overridable via command-line args) ─────────────────────
STACK_PATH = Path("coin_captures_ids_3250/stack.npy")
DOWNSAMPLE = 1   # spatially average NxN pixel patches (1 = no downsampling)

# Command-line overrides. parse_known_args so this stays usable when run cell-by-
# cell (e.g. in an interactive/notebook session where extra argv may be present).
_parser = argparse.ArgumentParser(description="OCT depth map from a captured Z-stack.")
_parser.add_argument("-n", "--downsample", type=int, default=DOWNSAMPLE,
                     help=f"spatially average NxN pixel patches (default {DOWNSAMPLE}, 1 = off)")
_parser.add_argument("-s", "--stack", type=Path, default=STACK_PATH,
                     help=f"path to the stack .npy (default {STACK_PATH})")
_args, _ = _parser.parse_known_args()
DOWNSAMPLE = _args.downsample
STACK_PATH = _args.stack

# ──────────────────────────────────────────────────────────────────────────────


def downsample_spatial(arr, n):
    """Average non-overlapping NxN patches over the first two (spatial) axes.

    arr is (H, W, ...) — extra trailing axes (e.g. frames) are preserved. The
    height/width are cropped to a multiple of n before reshaping into blocks.
    """
    if n <= 1:
        return arr
    h, w = arr.shape[:2]
    h2, w2 = (h // n) * n, (w // n) * n
    arr = arr[:h2, :w2]
    blocks = arr.reshape((h2 // n, n, w2 // n, n) + arr.shape[2:])
    return blocks.mean(axis=(1, 3))


#%%
frames = np.load(STACK_PATH)
print(f"Loaded: {STACK_PATH}  shape={frames.shape}")

if DOWNSAMPLE > 1:
    frames = downsample_spatial(frames, DOWNSAMPLE)
    print(f"Downsampled {DOWNSAMPLE}x{DOWNSAMPLE} -> shape={frames.shape}")

_, md_image, _ = compute_mean_diff(frames[:, :, ::2], patch_size=5, avg_type="local", temporal_window=20)

depth  = np.argmax(md_image, axis=2)
maxamp = np.max(md_image, axis=2)

# Tag outputs with the downsample factor so runs don't overwrite each other.
DS_TAG = f"_ds{DOWNSAMPLE}" if DOWNSAMPLE > 1 else ""

np.save(STACK_PATH.with_name(STACK_PATH.stem + DS_TAG + "_depth.npy"),  depth)
np.save(STACK_PATH.with_name(STACK_PATH.stem + DS_TAG + "_maxamp.npy"), maxamp)
print(f"Saved: {STACK_PATH.stem}{DS_TAG}_depth.npy  {STACK_PATH.stem}{DS_TAG}_maxamp.npy")
print(f"Depth range: {depth.min()} – {depth.max()} frames")

DEPTH_PATH = STACK_PATH.with_name(STACK_PATH.stem + DS_TAG + "_depth.npy")
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
ax.set_title("Depth map" + (f"  ({DOWNSAMPLE}x{DOWNSAMPLE} downsampled)" if DOWNSAMPLE > 1 else ""))
ax.axis("off")
plt.tight_layout()

out = DEPTH_PATH.with_name(DEPTH_PATH.stem + "_render.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
