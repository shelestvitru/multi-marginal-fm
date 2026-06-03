"""Compare linear vs piecewise-linear vs cubic spline interpolation from N(0,1) noise to clean image."""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from skimage.segmentation import slic
from scipy.interpolate import CubicSpline

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

levels = [
    {"n_segments": 64, "sigma": 2},
    {"n_segments": 16, "sigma": 4},
    {"n_segments": 4, "sigma": 6},
]


def superpixel_degrade(img_np, n_segments, sigma):
    segments = slic(img_np, n_segments=n_segments, sigma=sigma, start_label=0, channel_axis=-1)
    out = np.zeros_like(img_np, dtype=np.float64)
    for seg_id in range(segments.max() + 1):
        mask = segments == seg_id
        out[mask] = img_np[mask].mean(axis=0)
    return np.clip(out, 0, 1).astype(np.float32)


rng = np.random.default_rng(42)
indices = rng.choice(len(dataset), size=5, replace=False)

# Observed time points for cubic: noise -> seg4 -> seg16 -> seg64 -> clean
t_observed = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

n_frames = 13
t_interp = np.linspace(0, 1, n_frames)

# Three rows per image: linear, piecewise-linear, cubic
n_images = 5
n_methods = 3
fig, axes = plt.subplots(n_images * n_methods, n_frames, figsize=(n_frames * 1.2, n_images * n_methods * 1.3))

for img_i, idx in enumerate(indices):
    img_pil, label = dataset[idx]
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    # Fixed noise sample for this image
    noise = rng.standard_normal(img_np.shape).astype(np.float32)

    # Degradation levels in restoration order (most degraded first)
    deg4 = superpixel_degrade(img_np, n_segments=4, sigma=6)
    deg16 = superpixel_degrade(img_np, n_segments=16, sigma=4)
    deg64 = superpixel_degrade(img_np, n_segments=64, sigma=2)

    observed = np.stack([noise, deg4, deg16, deg64, img_np])  # (5, H, W, C)

    # --- Linear: noise -> clean ---
    row_lin = img_i * n_methods
    for col, t in enumerate(t_interp):
        frame = (1 - t) * noise + t * img_np
        axes[row_lin, col].imshow(np.clip(frame, 0, 1))
        axes[row_lin, col].axis("off")

    # --- Piecewise linear: noise -> seg4 -> seg16 -> seg64 -> clean ---
    row_pwl = img_i * n_methods + 1
    for col, t in enumerate(t_interp):
        # Find which segment t falls in
        seg_idx = np.searchsorted(t_observed, t, side="right") - 1
        seg_idx = np.clip(seg_idx, 0, len(t_observed) - 2)
        t_local = (t - t_observed[seg_idx]) / (t_observed[seg_idx + 1] - t_observed[seg_idx])
        frame = (1 - t_local) * observed[seg_idx] + t_local * observed[seg_idx + 1]
        axes[row_pwl, col].imshow(np.clip(frame, 0, 1))
        axes[row_pwl, col].axis("off")

    # --- Cubic: noise -> seg4 -> seg16 -> seg64 -> clean ---
    flat = observed.reshape(5, -1)
    cs = CubicSpline(t_observed, flat, axis=0)
    interp_flat = cs(t_interp)
    interp_imgs = interp_flat.reshape(n_frames, *img_np.shape)

    row_cub = img_i * n_methods + 2
    for col in range(n_frames):
        axes[row_cub, col].imshow(np.clip(interp_imgs[col], 0, 1))
        axes[row_cub, col].axis("off")

    # Row labels
    class_name = CIFAR10_CLASSES[label]
    axes[row_lin, 0].set_ylabel(f"{class_name}\nlinear", fontsize=7, rotation=0, labelpad=40, va="center")
    axes[row_lin, 0].set_xticks([])
    axes[row_lin, 0].set_yticks([])
    axes[row_pwl, 0].set_ylabel("pw-linear", fontsize=7, rotation=0, labelpad=40, va="center")
    axes[row_pwl, 0].set_xticks([])
    axes[row_pwl, 0].set_yticks([])
    axes[row_cub, 0].set_ylabel("cubic", fontsize=7, rotation=0, labelpad=40, va="center")
    axes[row_cub, 0].set_xticks([])
    axes[row_cub, 0].set_yticks([])

# Column titles
for col in range(n_frames):
    t = t_interp[col]
    is_observed = np.any(np.abs(t_observed - t) < 1e-6)
    marker = " *" if is_observed else ""
    axes[0, col].set_title(f"t={t:.2f}{marker}", fontsize=7)

plt.suptitle("Linear vs Piecewise-Linear vs Cubic: N(0,1) → clean image (* = observed marginals)", fontsize=10, y=1.01)
plt.tight_layout()
from pathlib import Path
Path("out").mkdir(exist_ok=True)
plt.savefig("out/linear_vs_cubic.png", dpi=150, bbox_inches="tight")
print("Saved out/linear_vs_cubic.png")
