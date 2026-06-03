"""UNet for conditional flow matching on CIFAR-10 (32x32) with time + class conditioning."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConditioningMLP(nn.Module):
    """Projects time embedding + class embedding + optional segment embedding into conditioning vector."""

    def __init__(self, time_dim, num_classes, cond_dim, num_segments=0):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.segment_emb = nn.Embedding(num_segments, cond_dim) if num_segments > 0 else None

    def forward(self, t, y, seg_idx=None):
        cond = self.time_mlp(t) + self.class_emb(y)
        if self.segment_emb is not None and seg_idx is not None:
            cond = cond + self.segment_emb(seg_idx)
        return cond


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        # Add conditioning (broadcast over spatial dims)
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = self.norm(x).reshape(b, c, h * w).permute(0, 2, 1)  # (B, HW, C)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        return x + attn_out.permute(0, 2, 1).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for 32x32 images.

    Architecture:
        - 4 resolution levels: 32 -> 16 -> 8 -> 4
        - Channel multipliers: [1, 2, 2, 2] * base_channels
        - Self-attention at 16x16
        - 2 ResBlocks per level
        - Time + class conditioning via additive projection
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(16,),
        num_classes=10,
        num_segments=0,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        cond_dim = base_channels * 4
        time_dim = base_channels

        # Conditioning
        self.cond_mlp = ConditioningMLP(time_dim, num_classes, cond_dim, num_segments=num_segments)

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        # Each "unit" = ResBlock + optional Attention, producing one skip
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = [base_channels]  # track skip channel sizes
        ch = base_channels
        res = 32

        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_res.append(ResBlock(ch, out_ch, cond_dim, dropout))
                self.down_attn.append(
                    AttentionBlock(out_ch) if res in attn_resolutions else None
                )
                ch = out_ch
                channels.append(ch)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(ch))
                channels.append(ch)
                res //= 2
            else:
                self.downsamplers.append(None)

        # Store how many res blocks per level for the forward pass
        self._num_res_blocks = num_res_blocks
        self._num_levels = len(channel_mults)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, cond_dim, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, cond_dim, dropout)

        # Decoder (mirrors encoder)
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                self.up_res.append(ResBlock(ch + skip_ch, out_ch, cond_dim, dropout))
                self.up_attn.append(
                    AttentionBlock(out_ch) if res in attn_resolutions else None
                )
                ch = out_ch
            if level > 0:
                self.upsamplers.append(Upsample(ch))
                res *= 2
            else:
                self.upsamplers.append(None)

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, t, y, seg_idx=None):
        """
        Args:
            x: (B, C, H, W) noisy image
            t: (B,) time in [0, 1]
            y: (B,) integer class labels
            seg_idx: (B,) optional segment index for multi-marginal models
        Returns:
            (B, C, H, W) predicted velocity
        """
        cond = self.cond_mlp(t, y, seg_idx)

        h = self.input_conv(x)
        skips = [h]

        # Encoder
        block_idx = 0
        for level in range(self._num_levels):
            for _ in range(self._num_res_blocks):
                h = self.down_res[block_idx](h, cond)
                if self.down_attn[block_idx] is not None:
                    h = self.down_attn[block_idx](h)
                skips.append(h)
                block_idx += 1
            if self.downsamplers[level] is not None:
                h = self.downsamplers[level](h)
                skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder
        block_idx = 0
        for rev_level, level in enumerate(reversed(range(self._num_levels))):
            for _ in range(self._num_res_blocks + 1):
                h = self.up_res[block_idx](torch.cat([h, skips.pop()], dim=1), cond)
                if self.up_attn[block_idx] is not None:
                    h = self.up_attn[block_idx](h)
                block_idx += 1
            if self.upsamplers[rev_level] is not None:
                h = self.upsamplers[rev_level](h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


if __name__ == "__main__":
    model = UNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params / 1e6:.1f}M")

    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    out = model(x, t, y)
    print(f"Input: {x.shape} -> Output: {out.shape}")
