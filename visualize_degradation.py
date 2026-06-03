"""Visualize superpixel degradation chain on CIFAR-10 images."""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
from skimage.segmentation import slic

# Download CIFAR-10
dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)

# Pick 10 random images
rng = np.random.default_rng(42)
indices = rng.choice(len(dataset), size=10, replace=False)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Degradation parameters: 3 levels
levels = [
    {"n_segments": 64, "sigma": 2},
    {"n_segments": 16, "sigma": 4},
    {"n_segments": 4, "sigma": 6},
]


def superpixel_degrade(img_np, n_segments, sigma):
    """Apply SLIC superpixel segmentation, replacing each segment with its mean color."""
    segments = slic(img_np, n_segments=n_segments, sigma=sigma, start_label=0, channel_axis=-1)
    out = np.zeros_like(img_np, dtype=np.float64)
    for seg_id in range(segments.max() + 1):
        mask = segments == seg_id
        out[mask] = img_np[mask].mean(axis=0)
    return np.clip(out, 0, 1).astype(np.float32)


# Build grid: 10 rows x 4 columns (original + 3 degradation levels)
n_cols = 1 + len(levels)
fig, axes = plt.subplots(10, n_cols, figsize=(n_cols * 1.8, 10 * 1.8))

col_titles = ["Original"] + [f"seg={l['n_segments']}, σ={l['sigma']}" for l in levels]

for row, idx in enumerate(indices):
    img_pil, label = dataset[idx]
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    axes[row, 0].imshow(img_np)
    axes[row, 0].set_ylabel(CIFAR10_CLASSES[label], fontsize=9, rotation=0, labelpad=40, va="center")
    axes[row, 0].set_xticks([])
    axes[row, 0].set_yticks([])

    for col, params in enumerate(levels, start=1):
        degraded = superpixel_degrade(img_np, params["n_segments"], params["sigma"])
        axes[row, col].imshow(degraded)
        axes[row, col].axis("off")

# Column titles
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=9)

plt.suptitle("CIFAR-10 Superpixel Degradation Chain", fontsize=13, y=1.01)
plt.tight_layout()
from pathlib import Path
Path("out").mkdir(exist_ok=True)
plt.savefig("out/degradation_grid.png", dpi=150, bbox_inches="tight")
print("Saved out/degradation_grid.png")
