"""Visualize cubic spline interpolation between superpixel degradation levels."""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from skimage.segmentation import slic
from scipy.interpolate import CubicSpline

# Download CIFAR-10
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Degradation parameters
levels = [
    {"n_segments": 64, "sigma": 2},
    {"n_segments": 16, "sigma": 4},
    {"n_segments": 4, "sigma": 6},
]

# Observed time points: original at t=0, then 3 degradation levels
t_observed = np.array([0.0, 1 / 3, 2 / 3, 1.0])


def superpixel_degrade(img_np, n_segments, sigma):
    segments = slic(img_np, n_segments=n_segments, sigma=sigma, start_label=0, channel_axis=-1)
    out = np.zeros_like(img_np, dtype=np.float64)
    for seg_id in range(segments.max() + 1):
        mask = segments == seg_id
        out[mask] = img_np[mask].mean(axis=0)
    return np.clip(out, 0, 1).astype(np.float32)


# Pick 5 images (fewer rows since we have many columns now)
rng = np.random.default_rng(42)
indices = rng.choice(len(dataset), size=5, replace=False)

# Interpolation times: dense sampling between 0 and 1
n_interp = 13  # total frames including endpoints
t_interp = np.linspace(0, 1, n_interp)

fig, axes = plt.subplots(5, n_interp, figsize=(n_interp * 1.2, 5 * 1.3))

for row, idx in enumerate(indices):
    img_pil, label = dataset[idx]
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    # Compute observed degradation levels
    observed = [img_np]  # t=0: original
    for params in levels:
        observed.append(superpixel_degrade(img_np, params["n_segments"], params["sigma"]))
    observed = np.stack(observed)  # (4, H, W, C)

    # Flatten spatial dims, fit cubic spline per pixel
    flat = observed.reshape(4, -1)  # (4, H*W*C)
    cs = CubicSpline(t_observed, flat, axis=0)

    # Evaluate at all interpolation times
    interp_flat = cs(t_interp)  # (n_interp, H*W*C)
    interp_imgs = interp_flat.reshape(n_interp, *img_np.shape)
    interp_imgs = np.clip(interp_imgs, 0, 1)

    for col in range(n_interp):
        axes[row, col].imshow(interp_imgs[col])
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=8, rotation=0, labelpad=35, va="center")
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

# Mark observed vs interpolated times in column titles
for col in range(n_interp):
    t = t_interp[col]
    is_observed = np.any(np.abs(t_observed - t) < 1e-6)
    marker = " *" if is_observed else ""
    axes[0, col].set_title(f"t={t:.2f}{marker}", fontsize=7)

plt.suptitle("Cubic Spline Interpolation (* = observed marginals)", fontsize=11, y=1.01)
plt.tight_layout()
from pathlib import Path
Path("out").mkdir(exist_ok=True)
plt.savefig("out/interpolation_grid.png", dpi=150, bbox_inches="tight")
print("Saved out/interpolation_grid.png")
