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
    """Compute per-frame interference amplitude image (depth-from-focus measure).

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
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save_colormap(array: np.ndarray, path: Path, cmap: str = "viridis", title: str = ""):
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap=cmap)
    plt.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main: load stack → compute → save
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute depth-from-focus map from a .npy frame stack.")
    parser.add_argument("stack",           type=str,   help="Path to input .npy stack (H x W x N).")
    parser.add_argument("--patch-size",    type=int,   default=5,       help="Spatial patch size (default 5).")
    parser.add_argument("--avg-type",      type=str,   default="local",  help="'local' or 'global' (default local).")
    parser.add_argument("--temporal-window", type=int, default=20,      help="Temporal window for local averaging (default 20).")
    parser.add_argument("--every-other",   action="store_true",         help="Use only every other frame (frames[:,:,::2]).")
    args = parser.parse_args()

    stack_path = Path(args.stack)
    out_dir = stack_path.parent
    stem = stack_path.stem

    print(f"Loading: {stack_path}")
    frames = np.load(stack_path)                          # (H, W, N)
    print(f"  shape={frames.shape}  dtype={frames.dtype}")

    if args.every_other:
        frames = frames[:, :, ::2]
        print(f"  every-other frame → shape={frames.shape}")

    print(f"Running compute_mean_diff  patch={args.patch_size}  type={args.avg_type}  window={args.temporal_window}")
    _, md_image, _ = compute_mean_diff(
        frames,
        patch_size=args.patch_size,
        avg_type=args.avg_type,
        temporal_window=args.temporal_window,
    )

    maxamp = np.max(md_image, axis=2)                     # (H, W)
    depth  = np.argmax(md_image, axis=2)                  # (H, W)  0-indexed frame number

    # Save raw arrays
    np.save(out_dir / f"{stem}_depth.npy",  depth)
    np.save(out_dir / f"{stem}_maxamp.npy", maxamp)
    print(f"Saved: {stem}_depth.npy  {stem}_maxamp.npy")

    # Save visualisations
    _save_colormap(depth,  out_dir / f"{stem}_depth.png",  cmap="viridis", title="Depth map")
    _save_colormap(maxamp, out_dir / f"{stem}_maxamp.png", cmap="inferno", title="Max amplitude")
    print(f"Saved: {stem}_depth.png  {stem}_maxamp.png")
