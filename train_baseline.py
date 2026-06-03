"""
Baseline CFM training on CIFAR-10 with linear interpolation.

Flow direction: N(0,1) noise (t=0) -> clean image (t=1)
Interpolation: x_t = (1-t)*noise + t*clean
Velocity target: v = clean - noise

Usage:
    python train_baseline.py --run_name baseline_v1
    tensorboard --logdir runs/
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
import torchvision

from model import UNet

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────


class CIFAR10_Normalized(Dataset):
    """CIFAR-10 normalized to [-1, 1]."""

    def __init__(self, train=True):
        self.dataset = torchvision.datasets.CIFAR10(root="./data", train=train, download=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # (3, 32, 32)
        return img, label


# ──────────────────────────────────────────────────────────────
# Sampling (ODE integration)
# ──────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_trajectories(model, device, n_samples=10, n_steps=100, n_trajectory_frames=13):
    model.eval()
    classes = torch.arange(n_samples, device=device) % 10
    x = torch.randn(n_samples, 3, 32, 32, device=device)

    dt = 1.0 / n_steps
    save_steps = set(np.linspace(0, n_steps, n_trajectory_frames, dtype=int).tolist())
    trajectories = []

    for step in range(n_steps + 1):
        if step in save_steps:
            trajectories.append(x.clone())
        if step < n_steps:
            t = torch.full((n_samples,), step * dt, device=device)
            v = model(x, t, classes)
            x = x + v * dt

    trajectories = torch.stack(trajectories, dim=1)
    model.train()
    return trajectories, classes


def log_trajectory_grid(writer, trajectories, classes, global_step):
    import matplotlib.pyplot as plt

    n_samples, n_frames = trajectories.shape[:2]
    fig, axes = plt.subplots(n_samples, n_frames, figsize=(n_frames * 1.5, n_samples * 1.5))

    for i in range(n_samples):
        for j in range(n_frames):
            img = trajectories[i, j].cpu().permute(1, 2, 0).numpy()
            img = np.clip((img + 1) / 2, 0, 1)
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

    plt.suptitle(f"Baseline CFM trajectories (step {global_step})", fontsize=10)
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
        args.run_name = f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    dataset = CIFAR10_Normalized(train=True)
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

        for clean, labels in loader:
            clean = clean.to(device)
            labels = labels.to(device)

            # Sample noise and time
            noise = torch.randn_like(clean)
            t = torch.rand(clean.shape[0], device=device)

            # Linear interpolation: x_t = (1-t)*noise + t*clean
            t_expand = t[:, None, None, None]
            x_t = (1 - t_expand) * noise + t_expand * clean

            # Target velocity: v = clean - noise
            v_target = clean - noise

            # Forward pass
            v_pred = model(x_t, t, labels)

            # MSE loss
            loss = (v_pred - v_target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            writer.add_scalar("loss/step", loss.item(), global_step)

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
