import numpy as np
from scipy.ndimage import uniform_filter, median_filter
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────────
DEPTH_PATH   = Path("coin_captures/stack_depth.npy")

# Local-variance outlier removal:
# pixels whose neighbourhood std exceeds this threshold are discarded
OUTLIER_WINDOW    = 5      # window size for local std computation
OUTLIER_STD_MAX   = 3.0    # max allowed local std (in frame units) — lower = stricter

# Optional: smooth the surviving depth values before meshing
SMOOTH_AFTER      = True
SMOOTH_SIZE       = 3      # median filter size applied after outlier removal

# Depth → Z axis scale  (1 frame = how many pixels in XY?)
# increase to exaggerate height, decrease to flatten
Z_SCALE      = 2.0

OUTPUT_PATH  = DEPTH_PATH.with_suffix(".ply")
# ──────────────────────────────────────────────────────────────────────────────


def local_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Per-pixel standard deviation within a (size × size) window."""
    mean    = uniform_filter(arr.astype(np.float64), size=size)
    mean_sq = uniform_filter(arr.astype(np.float64) ** 2, size=size)
    return np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))


def save_ply_mesh(path: Path, depth: np.ndarray, mask: np.ndarray, z_scale: float):
    """Write a PLY mesh. mask=True means the pixel is VALID."""
    H, W = depth.shape

    # Assign a vertex index to each valid pixel (-1 = invalid)
    idx = np.full((H, W), -1, dtype=np.int32)
    valid_yx = np.argwhere(mask)
    idx[mask] = np.arange(len(valid_yx))

    vertices = np.column_stack([
        valid_yx[:, 1].astype(np.float32),          # x = col
        valid_yx[:, 0].astype(np.float32),          # y = row
        depth[mask].astype(np.float32) * z_scale,   # z = scaled depth
    ])

    # Build faces from 2×2 quads — only include quads where all 4 corners are valid
    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            tl = idx[r,   c  ]
            tr = idx[r,   c+1]
            bl = idx[r+1, c  ]
            br = idx[r+1, c+1]
            if tl < 0 or tr < 0 or bl < 0 or br < 0:
                continue
            faces.append((tl, tr, bl))
            faces.append((tr, br, bl))

    faces = np.array(faces, dtype=np.int32)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved: {path}  ({len(vertices)} vertices, {len(faces)} faces)")


# ── Load ──────────────────────────────────────────────────────────────────────
depth = np.load(DEPTH_PATH).astype(np.float32)
print(f"Loaded : {DEPTH_PATH}  shape={depth.shape}  range={depth.min():.0f}–{depth.max():.0f}")

# ── Outlier detection: high local variance = unreliable ───────────────────────
std_map = local_std(depth, size=OUTLIER_WINDOW)
outlier_mask = std_map > OUTLIER_STD_MAX
valid_mask   = ~outlier_mask

n_removed = outlier_mask.sum()
print(f"Outliers removed: {n_removed} / {depth.size} px  ({100*n_removed/depth.size:.1f}%)  "
      f"[local std > {OUTLIER_STD_MAX} in {OUTLIER_WINDOW}×{OUTLIER_WINDOW} window]")

# ── Optional smooth on surviving pixels ───────────────────────────────────────
if SMOOTH_AFTER:
    smoothed = median_filter(depth, size=SMOOTH_SIZE)
    depth[valid_mask] = smoothed[valid_mask]
    print(f"Smoothed surviving pixels with {SMOOTH_SIZE}×{SMOOTH_SIZE} median filter")

# ── Write PLY ─────────────────────────────────────────────────────────────────
save_ply_mesh(OUTPUT_PATH, depth, valid_mask, Z_SCALE)
print(f"\nOpen in MeshLab:  File → Import Mesh → {OUTPUT_PATH.name}")
print(f"Tip: in MeshLab, Filters → Smoothing → Laplacian Smooth for a nicer surface")
