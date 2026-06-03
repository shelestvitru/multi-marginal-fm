"""Inference: load checkpoint from a run, sample trajectories, save grid.

Usage:
    python inference.py --run_name my_experiment                    # last checkpoint in run
    python inference.py --ckpt runs/my_experiment/checkpoints/epoch0500.pt  # specific checkpoint
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model import UNet

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

N_SNAPSHOTS = 9
SNAPSHOT_TIMES = np.linspace(0, 1, N_SNAPSHOTS).tolist()


# Multi-marginal time config (must match train.py)
T_OBSERVED = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
N_SEGMENTS = len(T_OBSERVED) - 1


@torch.no_grad()
def sample(model, device, n_steps=100, use_segments=False):
    """Euler-integrate the velocity field from t=0 to t=1 for all 10 classes."""
    model.eval()
    n_samples = 10
    classes = torch.arange(n_samples, device=device)
    x = torch.randn(n_samples, 3, 32, 32, device=device)

    dt = 1.0 / n_steps
    snap_steps = {round(t * n_steps): t for t in SNAPSHOT_TIMES}
    snapshots = {}

    for step in range(n_steps + 1):
        if step in snap_steps:
            snapshots[snap_steps[step]] = x.clone()
        if step < n_steps:
            global_t = step * dt

            if use_segments:
                # Map global time -> segment index + local time
                seg = min(int(global_t * N_SEGMENTS), N_SEGMENTS - 1)
                t_left = T_OBSERVED[seg]
                t_right = T_OBSERVED[seg + 1]
                t_local_val = max(0.0, min(1.0, (global_t - t_left) / (t_right - t_left)))

                seg_idx = torch.full((n_samples,), seg, device=device, dtype=torch.long)
                t_local = torch.full((n_samples,), t_local_val, device=device)
                v = model(x, t_local, classes, seg_idx=seg_idx)
            else:
                t = torch.full((n_samples,), global_t, device=device)
                v = model(x, t, classes)

            x = x + v * dt

    return snapshots, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Run name (looks in runs/<run_name>/checkpoints/)")
    parser.add_argument("--ckpt", type=str, default=None, help="Direct checkpoint path")
    cli_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint and run directory
    if cli_args.ckpt:
        ckpt_path = Path(cli_args.ckpt)
        # Derive run_dir if checkpoint is inside a run directory
        if ckpt_path.parent.name == "checkpoints":
            run_dir = ckpt_path.parent.parent
        else:
            run_dir = None
    elif cli_args.run_name:
        run_dir = Path("runs") / cli_args.run_name
        ckpt_files = sorted(run_dir.glob("checkpoints/epoch*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {run_dir / 'checkpoints'}")
        ckpt_path = ckpt_files[-1]
    else:
        raise ValueError("Provide --run_name or --ckpt")

    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    # Read run name from config if available
    if run_dir and (run_dir / "config.json").exists():
        run_config = json.loads((run_dir / "config.json").read_text())
        model_name = run_config.get("run_name", run_dir.name)
    else:
        model_name = run_dir.name if run_dir else ckpt_path.stem

    # Detect if model uses segment embedding by checking state dict
    use_segments = any("segment_emb" in k for k in ckpt["model"].keys())
    num_segments = N_SEGMENTS if use_segments else 0

    model = UNet(
        base_channels=args.get("base_channels", 128),
        num_classes=10,
        num_segments=num_segments,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"Segment conditioning: {use_segments}")

    snapshots, classes = sample(model, device, use_segments=use_segments)

    # Build grid: rows = classes, cols = snapshot times
    n_rows = 10
    n_cols = len(SNAPSHOT_TIMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))

    for row in range(n_rows):
        for col, t in enumerate(SNAPSHOT_TIMES):
            img = snapshots[t][row].cpu().permute(1, 2, 0).numpy()
            img = np.clip((img + 1) / 2, 0, 1)  # [-1, 1] -> [0, 1]
            axes[row, col].imshow(img)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        class_idx = classes[row].item()
        axes[row, 0].set_ylabel(
            f"{class_idx}: {CIFAR10_CLASSES[class_idx]}",
            fontsize=8, rotation=0, labelpad=50, va="center",
        )

    for col, t in enumerate(SNAPSHOT_TIMES):
        axes[0, col].set_title(f"t={t:.2f}", fontsize=9)

    plt.suptitle(f"{model_name} (epoch {ckpt['epoch']})", fontsize=11)
    plt.tight_layout()

    # Save to run's outputs dir if available, otherwise to out/
    if run_dir:
        out_dir = run_dir / "outputs"
    else:
        out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inference_epoch{ckpt['epoch']:04d}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
