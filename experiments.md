# Multi-Marginal Flow Matching for Images: Experiment Log

## Goal

Apply the ALI-FM (Adversarially Learned Interpolants) multi-marginal flow matching framework to image generation on CIFAR-10 (32x32). The original paper targets synthetic/biological data with shallow MLPs. We want to see if the multi-marginal approach can work for images and whether it offers advantages over standard two-point flow matching.

## Degradation Chain

We use SLIC superpixel segmentation as a deterministic degradation, producing paired intermediate marginals:

| Time | Marginal | Parameters |
|------|----------|------------|
| t=0 | N(0,1) noise | Standard Gaussian |
| t=0.25 | seg4 | n_segments=4, sigma=6 |
| t=0.50 | seg16 | n_segments=16, sigma=4 |
| t=0.75 | seg64 | n_segments=64, sigma=2 |
| t=1.0 | clean | Original CIFAR-10 image |

The flow direction is noise → clean (generative). Since degradation is deterministic per-image, we have exact per-sample pairing across all marginals, which means we can skip the GAN stage and OT coupling from the original paper.

Precomputed degradations are saved to `data/precomputed/` (~1.4GB for 50K images x 3 levels).

See `out/degradation_grid.png` for visualization of the degradation chain and `out/linear_vs_cubic.png` for comparison of linear vs piecewise-linear vs cubic interpolation paths.

## Architecture

UNet (35.8M parameters) for all experiments:
- Base channels: 128, multipliers: [1, 2, 2, 2]
- Resolutions: 32 → 16 → 8 → 4
- Self-attention at 16x16
- 2 ResBlocks per level
- Time conditioning: sinusoidal embedding → MLP
- Class conditioning: nn.Embedding(10, cond_dim), additive
- Dropout: 0.1

## Shared Training Setup

- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- No LR schedule (constant lr)
- Gradient clipping: 1.0
- Batch size: 2048
- Hardware: NVIDIA H200
- Data normalized to [-1, 1]
- Sampling: Euler integration, 100 steps from t=0 to t=1

## Experiments

### 1. Baseline CFM (linear interpolation)

**Run**: `runs/baseline/` (old format, no config.json) + `out/inference_baseline_epoch0550.png`

**Setup**: Standard conditional flow matching. Linear interpolation `x_t = (1-t)*noise + t*clean`, velocity target `v = clean - noise`. No intermediate marginals.

**Result at 550 epochs**: Good quality. Clean, recognizable class-specific images. Smooth denoising trajectory. This is our reference for what the UNet + training setup can achieve.

**Status**: Working well. This is the baseline to beat.

---

### 2. MM-FM with plain MSE loss (original attempt)

**Run**: `checkpoints/model_epoch2000.pt` (old format), `out/inference_mmfm_epoch2000.png`

**Setup**: Piecewise-linear interpolation through all 5 marginals. Plain MSE loss on velocity. Data in [0, 1] (not [-1, 1]). Cosine LR schedule.

**Result at 2000 epochs**: Model learns the first segment (noise→seg4) well — coarse color structure emerges by t=0.25. But later stages completely break down — images become noisy/fragmented after t=0.5 instead of sharpening.

**Diagnosis**: Two issues identified:
1. **Loss scale imbalance**: Velocity magnitude for noise→seg4 (~8) is ~10x larger than seg64→clean (~0.5). MSE loss is dominated by the first segment (gradient contribution ~100x larger), so the model underfits later segments.
2. **Exposure bias**: During training, the model only sees x_t on the exact piecewise-linear path. At inference, small errors at early steps compound — by t=0.5 the model sees states it never encountered, and velocity predictions become garbage. This is worse for MM-FM than baseline because the training distribution at each time is narrow and specific.

---

### 3. MM-FM with relative MSE loss + [-1,1] normalization

**Run**: `runs/mmfm_relloss_v1/`

**Config**: Same as #2 but with:
- Data normalized to [-1, 1] (matching noise scale better)
- Relative MSE loss: `loss = mean(MSE_per_sample / (v_target_magnitude^2 + eps))` to equalize gradient contribution across segments

**Result at 500 epochs**: No improvement. Same failure mode — model breaks down after the first segment. See `runs/mmfm_relloss_v1/outputs/inference_epoch0500.png`.

**Conclusion**: Loss reweighting alone doesn't fix the exposure bias problem. The model still only trains on the exact interpolation path, so inference drift is fatal.

---

### 4. MM-FM with adaptive variance schedule + relative MSE loss

**Run**: `runs/mmfm_adaptive_sigma_v1/`

**Config**: Added adaptive variance schedule from the paper:
- `sigma_t = M * sqrt(t_norm * (1 - t_norm))` within each segment
- `x_t = mu_t + sigma_t * epsilon` (adds noise to interpolated points)
- Velocity target updated to conditional form: `u_t = (sigma'_t/sigma_t)*(x_t - mu_t) + mu'_t`
- sigma_max = 0.5
- Still using relative MSE loss

**Result at 500 epochs**: Better than #2 and #3. Model now gets past the seg4 stage and produces recognizable (but noisy) images at t=1.0. The adaptive sigma helps with exposure bias by widening the training distribution. See `runs/mmfm_adaptive_sigma_v1/outputs/inference_epoch0500.png`.

**Conclusion**: Adaptive sigma is the right direction but quality still has a significant gap vs baseline. Relative loss may be interacting badly with the new velocity targets.

---

### 5. MM-FM with adaptive variance schedule + plain MSE loss

**Run**: `runs/mmfm_sigma_plainmse_v1/`

**Config**: Same as #4 but dropped the relative MSE loss, using plain MSE instead. Hypothesis: the relative loss normalization may conflict with the adaptive variance velocity targets.

**Result at 500 epochs**: Similar to #4, maybe slightly worse. Still noisy and lacking fine detail compared to baseline. See `runs/mmfm_sigma_plainmse_v1/outputs/inference_epoch0500.png`.

**Conclusion**: The loss function (relative vs plain MSE) isn't the main bottleneck. The adaptive sigma helps but isn't sufficient.

---

## Summary of What Works / Doesn't Work

| Fix | Impact |
|-----|--------|
| [-1, 1] normalization | Necessary but not sufficient |
| Relative MSE loss | No clear benefit |
| Adaptive variance schedule | Helps significantly — model now gets past first segment |
| More epochs (2000 vs 500) | Didn't help without adaptive sigma |

## Open Questions and Ideas for Next Steps

### Why is MM-FM harder than baseline?
The baseline learns ONE velocity field across a smooth, simple interpolation. MM-FM must learn a velocity field that changes character 4 times (one per segment), with different magnitudes and different types of transformations (denoising vs. structure refinement vs. detail recovery). The model has to partition its capacity across these very different tasks.

### Possible directions to explore:

1. **sigma_max tuning**: Current value 0.5 is arbitrary. Try 0.1, 0.3, 1.0. Too high drowns signal, too low doesn't fix exposure bias.

2. **Fewer intermediate marginals**: Maybe 5 points (4 segments) is too many. Try just noise → seg16 → clean (3 points, 2 segments). Fewer velocity discontinuities, simpler for the model.

3. **Different source distribution**: Instead of N(0,1), start from something closer to the image domain. E.g., sample mean color from the empirical distribution + small noise. Reduces the difficulty of the first segment. But this reduces diversity unless the source distribution is rich enough.

4. **Non-uniform time spacing**: Currently segments are equal width (0.25 each). The first segment (noise→seg4) is the hardest — give it more time. E.g., t=[0, 0.5, 0.7, 0.85, 1.0].

5. **Smooth interpolation instead of piecewise-linear**: Use cubic splines. Eliminates velocity discontinuities at knot points. The velocity field becomes continuous, which may be easier to learn. Downside: cubic overshoot (visible in our earlier visualization).

6. **Separate models per segment**: Train a different model (or different heads) for each segment. Each model only needs to learn one type of transformation. Expensive but diagnostic — if this works, the issue is capacity/multi-task competition.

7. **Progressive training**: Train segment 1 first until converged, then freeze and train segment 2, etc. Or curriculum learning — start with just 2 marginals (like baseline), then gradually add intermediate ones.

8. **Larger model**: Maybe 35M params isn't enough for the multi-task nature of MM-FM. The baseline succeeds because it has a simpler task. Try 2x base channels (128→256, ~140M params).

9. **Better ODE solver at inference**: Replace Euler with adaptive RK45 (e.g., torchdiffeq). Might help with accuracy near velocity discontinuities.

10. **Fundamentally different approach**: Instead of noise→clean with intermediate superpixel marginals, try clean→clean class-conditional transport with intermediate "style transfer" marginals. Or image inpainting with progressive mask reveal as the degradation chain.
