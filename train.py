"""
Multi-marginal CFM training on CIFAR-10 with piecewise-linear interpolation.

Flow direction: noise (t=0) -> seg4 (t=0.25) -> seg16 (t=0.5) -> seg64 (t=0.75) -> clean (t=1.0)

Usage:
    python precompute.py                           # run once
    python train.py --run_name my_experiment       # train
    tensorboard --logdir runs/                     # monitor
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import time

from model import UNet

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

# Observed time points for the piecewise-linear path
# noise -> seg4 -> seg16 -> seg64 -> clean
T_OBSERVED = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float32)
N_SEGMENTS = len(T_OBSERVED) - 1  # 4 linear segments

# Noise added to each marginal to balance velocity magnitudes across segments.
# Computed via optimization (sqrt-decay schedule).
# Index: [noise(implicit), seg4, seg16, seg64, clean]
MARGINAL_SIGMAS = np.array([1.0, 0.87, 0.71, 0.50, 0.0], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────


class MMFM_Dataset(Dataset):
    """
    Loads precomputed degradation levels.
    Each item returns all 5 marginals for one image + its class label.
    """

    def __init__(self, data_dir="data/precomputed", split="train"):
        data_dir = Path(data_dir)
        self.clean = np.load(data_dir / f"{split}_clean.npy")   # (N, 32, 32, 3)
        self.seg4 = np.load(data_dir / f"{split}_seg4.npy")
        self.seg16 = np.load(data_dir / f"{split}_seg16.npy")
        self.seg64 = np.load(data_dir / f"{split}_seg64.npy")
        self.labels = np.load(data_dir / f"{split}_labels.npy")  # (N,)
        self.n = len(self.labels)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return as CHW float32 tensors, normalized to [-1, 1]
        clean = torch.from_numpy(self.clean[idx]).permute(2, 0, 1) * 2 - 1
        seg4 = torch.from_numpy(self.seg4[idx]).permute(2, 0, 1) * 2 - 1
        seg16 = torch.from_numpy(self.seg16[idx]).permute(2, 0, 1) * 2 - 1
        seg64 = torch.from_numpy(self.seg64[idx]).permute(2, 0, 1) * 2 - 1
        label = int(self.labels[idx])
        return clean, seg4, seg16, seg64, label


# ──────────────────────────────────────────────────────────────
# Piecewise-linear interpolation
# ──────────────────────────────────────────────────────────────


def piecewise_linear_interpolate(t, marginals):
    """
    Given random t in [0, 1] and the 5 noisy marginals, compute x_t and target velocity.

    Each marginal already has per-marginal noise added (see MARGINAL_SIGMAS).
    The model receives local time within the segment [0, 1] and a segment index.

    Args:
        t: (B,) global times in [0, 1]
        marginals: list of 5 tensors each (B, 3, 32, 32), already with noise added

    Returns:
        x_t: (B, 3, 32, 32) interpolated point
        v_t: (B, 3, 32, 32) target velocity
        seg_idx: (B,) segment indices
        t_local: (B,) local time within segment [0, 1]
    """
    t_obs = torch.tensor(T_OBSERVED, device=t.device)
    B = t.shape[0]

    # Find which segment each t falls into
    seg_idx = torch.searchsorted(t_obs, t.contiguous(), right=True) - 1
    seg_idx = seg_idx.clamp(0, N_SEGMENTS - 1)

    # Local time within segment: t_local in [0, 1]
    t_left = t_obs[seg_idx]
    t_right = t_obs[seg_idx + 1]
    dt = t_right - t_left
    t_local = ((t - t_left) / dt).clamp(0, 1)

    # Gather left and right marginals for each sample
    marginals_stacked = torch.stack(marginals, dim=0)
    left = marginals_stacked[seg_idx, torch.arange(B)]
    right = marginals_stacked[seg_idx + 1, torch.arange(B)]

    # Interpolate
    t_local_4d = t_local[:, None, None, None]
    x_t = (1 - t_local_4d) * left + t_local_4d * right

    # Velocity = (right - left) / dt
    dt_4d = dt[:, None, None, None]
    v_t = (right - left) / dt_4d

    return x_t, v_t, seg_idx, t_local


# ──────────────────────────────────────────────────────────────
# Sampling (ODE integration)
# ──────────────────────────────────────────────────────────────

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def sample_trajectories(model, device, n_samples=8, n_steps=100, n_trajectory_frames=10):
    """
    Sample images by integrating the velocity field from t=0 to t=1 (Euler method).
    Maps global time to (segment_index, local_time) for the model.
    """
    model.eval()
    t_obs = torch.tensor(T_OBSERVED, device=device)
    classes = torch.arange(n_samples, device=device) % 10
    x = torch.randn(n_samples, 3, 32, 32, device=device)

    global_dt = 1.0 / n_steps
    save_steps = set(np.linspace(0, n_steps, n_trajectory_frames, dtype=int).tolist())
    trajectories = []

    for step in range(n_steps + 1):
        if step in save_steps:
            trajectories.append(x.clone())
        if step < n_steps:
            global_t = step * global_dt

            # Map global time -> segment index + local time
            seg = min(int(global_t * N_SEGMENTS), N_SEGMENTS - 1)
            t_left = T_OBSERVED[seg]
            t_right = T_OBSERVED[seg + 1]
            t_local_val = (global_t - t_left) / (t_right - t_left)
            t_local_val = max(0.0, min(1.0, t_local_val))

            seg_idx = torch.full((n_samples,), seg, device=device, dtype=torch.long)
            t_local = torch.full((n_samples,), t_local_val, device=device)

            v = model(x, t_local, classes, seg_idx=seg_idx)
            x = x + v * global_dt

    trajectories = torch.stack(trajectories, dim=1)
    model.train()
    return trajectories, classes


def log_trajectory_grid(writer, trajectories, classes, global_step):
    """Log trajectory grid to tensorboard: rows = samples, cols = time steps."""
    import matplotlib.pyplot as plt

    n_samples, n_frames = trajectories.shape[:2]
    fig, axes = plt.subplots(n_samples, n_frames, figsize=(n_frames * 1.5, n_samples * 1.5))

    for i in range(n_samples):
        for j in range(n_frames):
            img = trajectories[i, j].cpu().permute(1, 2, 0).numpy()
            img = np.clip((img + 1) / 2, 0, 1)  # [-1, 1] -> [0, 1]
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(
            CIFAR10_CLASSES[classes[i].item()],
            fontsize=8, rotation=0, labelpad=35, va="center",
        )
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

    for j in range(n_frames):
        t_val = j / (n_frames - 1)
        axes[0, j].set_title(f"t={t_val:.2f}", fontsize=7)

    plt.suptitle(f"Sampling trajectories (step {global_step})", fontsize=10)
    plt.tight_layout()
    writer.add_figure("samples/trajectories", fig, global_step)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Setup run directory
    if args.run_name is None:
        args.run_name = f"mmfm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path("runs") / args.run_name
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()
    (run_dir / "outputs").mkdir()
    print(f"Run directory: {run_dir}")

    # Save config
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Data
    dataset = MMFM_Dataset(split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # Model
    model = UNet(
        base_channels=args.base_channels,
        num_classes=10,
        num_segments=N_SEGMENTS,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.1f}M parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Logging
    writer = SummaryWriter(log_dir=str(run_dir / "tb"))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            clean, seg4, seg16, seg64, labels = [x.to(device) for x in batch]
            B = clean.shape[0]
            sigmas = torch.tensor(MARGINAL_SIGMAS, device=device)

            # Build noisy marginals: each marginal gets per-level noise added
            # t=0: pure noise (sigma=1.0 means it's just N(0,1))
            # t=0.25: seg4 + N(0, 0.87^2)
            # t=0.50: seg16 + N(0, 0.71^2)
            # t=0.75: seg64 + N(0, 0.50^2)
            # t=1.0: clean (no noise)
            noise_0 = torch.randn_like(clean) * sigmas[0]  # pure noise
            noisy_seg4 = seg4 + torch.randn_like(seg4) * sigmas[1]
            noisy_seg16 = seg16 + torch.randn_like(seg16) * sigmas[2]
            noisy_seg64 = seg64 + torch.randn_like(seg64) * sigmas[3]
            marginals = [noise_0, noisy_seg4, noisy_seg16, noisy_seg64, clean]

            # Random time for each sample
            t = torch.rand(B, device=device)

            # Get interpolated point, target velocity, segment index, and local time
            x_t, v_target, seg_idx, t_local = piecewise_linear_interpolate(t, marginals)

            # Forward pass: model receives local time + segment index
            v_pred = model(x_t, t_local, labels, seg_idx=seg_idx)

            # MSE loss
            loss = (v_pred - v_target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # Log per-step loss
            writer.add_scalar("loss/step", loss.item(), global_step)

        # Epoch stats
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} | lr={lr:.2e} | {elapsed:.1f}s")
        writer.add_scalar("loss/epoch", avg_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        # Sample trajectories every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            trajectories, classes = sample_trajectories(
                model, device, n_samples=10, n_steps=100, n_trajectory_frames=13,
            )
            log_trajectory_grid(writer, trajectories, classes, global_step)
            print(f"  -> Logged sample trajectories")

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0 or epoch == args.epochs:
            ckpt_path = ckpt_dir / f"epoch{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None, help="Unique run name (auto-generated if omitted)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(args)
