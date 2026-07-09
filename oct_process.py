"""OCT depth map from a captured Z-stack — thin driver around oct.py.

The pipeline (decimate → compute_mean_diff → argmax/max → save + sidecar) and
the rendering live in oct.py; this script just parameterizes them and keeps
the #%% cells for interactive use.
"""
from pathlib import Path
import argparse
import numpy as np
from oct import downsample_spatial, process_stack, save_depth_outputs, save_colormap

# ── Settings (defaults; overridable via command-line args) ─────────────────────
STACK_PATH   = Path("coin_captures_ids_3250/stack.npy")
DOWNSAMPLE   = 1   # spatially average NxN pixel patches (1 = no downsampling)
FRAME_STRIDE = 2   # captured frames per depth index (recorded in the sidecar)

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

#%%
frames = np.load(STACK_PATH)
print(f"Loaded: {STACK_PATH}  shape={frames.shape}")

if DOWNSAMPLE > 1:
    frames = downsample_spatial(frames, DOWNSAMPLE)
    print(f"Downsampled {DOWNSAMPLE}x{DOWNSAMPLE} -> shape={frames.shape}")

depth, maxamp = process_stack(frames, frame_stride=FRAME_STRIDE,
                              patch_size=5, avg_type="local", temporal_window=20)
print(f"Depth range: {depth.min()} – {depth.max()} frames")

# Saves <stem>[_dsN]_depth.npy/_maxamp.npy plus the _depth.json sidecar that
# records frame_stride/downsample for depth_to_pointcloud.py.
DEPTH_PATH = save_depth_outputs(
    depth, maxamp, STACK_PATH, frame_stride=FRAME_STRIDE, downsample=DOWNSAMPLE,
    params={"patch_size": 5, "avg_type": "local", "temporal_window": 20})

#%%
save_colormap(
    depth, DEPTH_PATH.with_name(DEPTH_PATH.stem + "_render.png"),
    cmap="viridis", colorbar_label="Depth (frame index)",
    title="Depth map" + (f"  ({DOWNSAMPLE}x{DOWNSAMPLE} downsampled)" if DOWNSAMPLE > 1 else ""),
    show=True)
