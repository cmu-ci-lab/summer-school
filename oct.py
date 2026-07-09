import numpy as np
from scipy.ndimage import uniform_filter, uniform_filter1d
from pathlib import Path
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Core functions (translated from getintAmplitude.m + computeMeanDiff.m)
# ──────────────────────────────────────────────────────────────────────────────

def get_int_amplitude(
    frames: np.ndarray,
    avg_intensity: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate per-frame interference amplitude.

    Args:
        frames:        (H, W, N) float array.
        avg_intensity: (H, W, N) temporally-blurred frames (the DC estimate).
        patch_size:    side length of the spatial averaging patch.

    Returns:
        interf_vector: (N,)      mean amplitude per frame over all pixels.
        interf_images: (H, W, N) spatially-smoothed amplitude image.
    """
    # AC component — absolute deviation from local mean
    interf_images = np.abs(frames.astype(np.float64) - avg_intensity)

    # scalar: mean amplitude per frame
    interf_vector = np.mean(interf_images, axis=(0, 1))

    # 2-D spatial box filter per frame (separable, matches MATLAB convn 'same')
    interf_images = uniform_filter(
        interf_images, size=(patch_size, patch_size, 1), mode="reflect"
    )

    return interf_vector, interf_images


def compute_mean_diff(
    frames: np.ndarray,
    patch_size: int = 10,
    avg_type: str = "local",
    temporal_window: int = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-frame interference amplitude image (OCT envelope measure).

    Args:
        frames:          (H, W, N) uint8, uint16, or float array.
        patch_size:      spatial patch size for amplitude estimation.
        avg_type:        'global' (mean of all frames) or 'local' (running average).
        temporal_window: length of the temporal box filter; required for 'local'.

    Returns:
        md_vector: (N,)       mean amplitude per frame.
        md_image:  (H, W, N)  per-pixel amplitude — use argmax over axis 2 for depth.
        blurred:   (H, W, N)  the DC (mean-intensity) estimate.
    """
    if avg_type == "global":
        blurred = np.mean(frames, axis=2, keepdims=True).repeat(frames.shape[2], axis=2)

    else:
        if temporal_window is None:
            raise ValueError("temporal_window must be set when avg_type='local'.")

        # convert integer types to float (matches MATLAB im2single / im2double)
        if frames.dtype == np.uint8:
            frames = frames.astype(np.float32)
        elif frames.dtype == np.uint16:
            frames = frames.astype(np.float64)

        # 1-D temporal box filter with zero-padding at boundaries (matches convn 'same')
        blurred = uniform_filter1d(
            frames.astype(np.float64),
            size=temporal_window,
            axis=2,
            mode="constant",
            cval=0.0,
        )

        # Replace zero-padded boundary frames with the nearest valid frame,
        # exactly as in the MATLAB code.
        half = temporal_window // 2
        blurred[:, :, :half]  = blurred[:, :, half : half + 1]
        blurred[:, :, -half:] = blurred[:, :, -(half + 1) : -(half)]

    md_vector, md_image = get_int_amplitude(frames, blurred, patch_size)
    return md_vector, md_image, blurred


# ──────────────────────────────────────────────────────────────────────────────
# Shared pipeline helpers (used by the CLI below, oct_process.py and oct_view.py)
# ──────────────────────────────────────────────────────────────────────────────

def downsample_spatial(arr: np.ndarray, n: int) -> np.ndarray:
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


def process_stack(frames: np.ndarray, frame_stride: int = 2, patch_size: int = 5,
                  avg_type: str = "local", temporal_window: int = 20,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Stack → (depth, maxamp). One place owns the decimation + argmax recipe.

    frame_stride decimates the stack (frames[:, :, ::stride]) before
    compute_mean_diff, so a depth index spans `frame_stride` captured frames.
    """
    if frame_stride > 1:
        frames = frames[:, :, ::frame_stride]
    _, md_image, _ = compute_mean_diff(frames, patch_size=patch_size,
                                       avg_type=avg_type,
                                       temporal_window=temporal_window)
    depth  = np.argmax(md_image, axis=2)   # (H, W)  0-indexed decimated frame no.
    maxamp = np.max(md_image, axis=2)      # (H, W)
    return depth, maxamp


def save_depth_outputs(depth: np.ndarray, maxamp: np.ndarray, stack_path: Path,
                       frame_stride: int, downsample: int = 1,
                       params: dict = None) -> Path:
    """Save <stem>[_dsN]_depth.npy, _maxamp.npy and a _depth.json sidecar.

    The sidecar records the geometry consumers need (frame_stride, downsample,
    source stack, maxamp filename) so tools like depth_to_pointcloud.py read it
    instead of re-deriving scan facts from filenames or hardcoded constants.
    Returns the depth .npy path.
    """
    import json
    stem = stack_path.stem + (f"_ds{downsample}" if downsample > 1 else "")
    depth_path  = stack_path.with_name(f"{stem}_depth.npy")
    maxamp_path = stack_path.with_name(f"{stem}_maxamp.npy")
    np.save(depth_path, depth)
    np.save(maxamp_path, maxamp)
    sidecar = {
        "source_stack": stack_path.name,
        "frame_stride": frame_stride,   # captured frames per depth index
        "downsample": downsample,       # lateral pixels per depth-map pixel
        "maxamp": maxamp_path.name,
        **(params or {}),
    }
    sidecar_path = depth_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"Saved: {depth_path.name}  {maxamp_path.name}  {sidecar_path.name}")
    return depth_path


def save_colormap(array: np.ndarray, path: Path, cmap: str = "viridis",
                  title: str = "", vmin=None, vmax=None, colorbar_label: str = "",
                  show: bool = False):
    """Render an array as a colormapped figure and save it (optionally show)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main: load stack → compute → save
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute OCT depth map from a .npy Z-stack.")
    parser.add_argument("stack",           type=str,   help="Path to input .npy stack (H x W x N).")
    parser.add_argument("--patch-size",    type=int,   default=5,       help="Spatial patch size (default 5).")
    parser.add_argument("--avg-type",      type=str,   default="local",  help="'local' or 'global' (default local).")
    parser.add_argument("--temporal-window", type=int, default=20,      help="Temporal window for local averaging (default 20).")
    parser.add_argument("--every-other",   action="store_true",         help="Use only every other frame (frames[:,:,::2]).")
    parser.add_argument("-n", "--downsample", type=int, default=1,
                        help="spatially average NxN pixel patches first (default 1 = off)")
    args = parser.parse_args()

    stack_path = Path(args.stack)

    print(f"Loading: {stack_path}")
    frames = np.load(stack_path)                          # (H, W, N)
    print(f"  shape={frames.shape}  dtype={frames.dtype}")

    if args.downsample > 1:
        frames = downsample_spatial(frames, args.downsample)
        print(f"  downsampled {args.downsample}x{args.downsample} → shape={frames.shape}")

    frame_stride = 2 if args.every_other else 1
    print(f"Running compute_mean_diff  patch={args.patch_size}  type={args.avg_type}  "
          f"window={args.temporal_window}  stride={frame_stride}")
    depth, maxamp = process_stack(frames, frame_stride=frame_stride,
                                  patch_size=args.patch_size,
                                  avg_type=args.avg_type,
                                  temporal_window=args.temporal_window)
    print(f"Depth range: {depth.min()} – {depth.max()} frames")

    depth_path = save_depth_outputs(
        depth, maxamp, stack_path, frame_stride=frame_stride,
        downsample=args.downsample,
        params={"patch_size": args.patch_size, "avg_type": args.avg_type,
                "temporal_window": args.temporal_window})

    # Save visualisations
    save_colormap(depth,  depth_path.with_suffix(".png"), cmap="viridis",
                  title="Depth map", colorbar_label="Depth (frame index)")
    save_colormap(maxamp, depth_path.with_name(depth_path.stem.replace("_depth", "_maxamp") + ".png"),
                  cmap="inferno", title="Max amplitude")
