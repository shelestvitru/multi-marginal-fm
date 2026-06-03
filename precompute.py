"""Precompute superpixel degradation levels for CIFAR-10 and save to disk."""

import numpy as np
import torchvision
from skimage.segmentation import slic
from pathlib import Path
from tqdm import tqdm

LEVELS = [
    {"n_segments": 4, "sigma": 6},
    {"n_segments": 16, "sigma": 4},
    {"n_segments": 64, "sigma": 2},
]

OUT_DIR = Path("data/precomputed")


def superpixel_degrade(img_np, n_segments, sigma):
    """Apply SLIC superpixel segmentation, replacing each segment with its mean color."""
    segments = slic(img_np, n_segments=n_segments, sigma=sigma, start_label=0, channel_axis=-1)
    out = np.zeros_like(img_np, dtype=np.float64)
    for seg_id in range(segments.max() + 1):
        mask = segments == seg_id
        out[mask] = img_np[mask].mean(axis=0)
    return np.clip(out, 0, 1).astype(np.float32)


def precompute_split(split_name, train, max_images=None):
    dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=True)
    n = min(len(dataset), max_images) if max_images else len(dataset)

    # Load all images as float32 [0, 1]
    images = np.stack([np.array(dataset[i][0]) for i in range(n)]).astype(np.float32) / 255.0
    labels = np.array([dataset[i][1] for i in range(n)], dtype=np.int64)

    print(f"\n{split_name}: {n} images, shape {images.shape}")

    # Compute degradation levels
    # Order: seg4 (most degraded) -> seg16 -> seg64 (least degraded)
    # This matches the flow direction: noise -> seg4 -> seg16 -> seg64 -> clean
    for level in LEVELS:
        name = f"seg{level['n_segments']}"
        degraded = np.zeros_like(images)
        for i in tqdm(range(n), desc=f"  {name}"):
            degraded[i] = superpixel_degrade(images[i], level["n_segments"], level["sigma"])

        out_path = OUT_DIR / f"{split_name}_{name}.npy"
        np.save(out_path, degraded)
        print(f"  Saved {out_path} ({degraded.nbytes / 1e6:.0f}MB)")

    # Save clean images and labels
    np.save(OUT_DIR / f"{split_name}_clean.npy", images)
    np.save(OUT_DIR / f"{split_name}_labels.npy", labels)
    print(f"  Saved clean images and labels")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images (for testing)")
    cli_args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precompute_split("train", train=True, max_images=cli_args.max_images)
    precompute_split("test", train=False, max_images=cli_args.max_images)
    print("\nDone!")
