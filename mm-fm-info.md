# Multi-Marginal Flow Matching with Adversarially Learned Interpolants

Reference implementation: https://github.com/mmacosha/adversarially-learned-interpolants

## Problem Setup

Standard flow matching learns a velocity field to transport a source distribution to a target distribution. It uses a fixed linear interpolation `x_t = (1-t)*x_0 + t*x_1` between the two endpoints to define training targets for the velocity field.

But what if you observe data at K > 2 time points? For example, cell populations measured at days 2, 3, 5, 7 — you have snapshots of the distribution at multiple times, and you want the learned flow to pass through *all* of them, not just the endpoints.

Naive two-point flow matching ignores intermediate marginals entirely. The learned flow will cut straight from source to target, with no guarantee that the distribution at intermediate times matches the observed data.

**Multi-marginal flow matching** solves this: learn a flow whose marginal distribution at each observed time t_k matches the observed distribution at that time.

## Core Idea: Learned Interpolants

Instead of using the fixed linear interpolation between endpoints, learn a *nonlinear interpolant* that respects all K marginals. Then train a standard velocity field on top of it.

The interpolant takes the form:

```
I_t(x_0, x_1) = (1-t)*x_0 + t*x_1 + t*(1-t)*C_theta(x_0, x_1, t)
```

Where `C_theta` is a neural network (the "correction") and the `t*(1-t)` factor is key: it guarantees boundary conditions `I_0 = x_0` and `I_1 = x_1` regardless of what the network outputs. The correction can only affect the path between endpoints, not the endpoints themselves.

## Three-Stage Training Pipeline

### Stage 1: Interpolant Pretraining (supervised MSE)

Warm-start the interpolant by supervised regression on observed intermediate marginals:

```
L_pretrain = E[||I_t(x_0, x_1) - x_t||^2]
```

where `(x_0, x_t, x_1)` are samples from the marginals at times `(0, t, T)`. This doesn't match distributions — it just gives a reasonable initialization so the GAN stage starts from a non-degenerate point.

### Stage 2: Adversarial Training (GAN)

The supervised MSE doesn't enforce distributional matching (the marginal of `I_t(x_0, x_1)` over random `x_0, x_1` may not match the observed distribution at time t). A discriminator is trained to distinguish:

- **Real**: samples from the observed marginal at time t
- **Fake**: samples from `I_t(x_0, x_1)` with `x_0, x_1` drawn from endpoint marginals

The generator (interpolant) loss combines adversarial loss with a regularization term:

```
L_gen = L_adversarial + lambda * L_regularization
```

**Why this stage is expensive**: It's a minimax game, not a regression. The discriminator and generator alternate updates, convergence is slow (~30K-70K steps in the reference implementation).

**When you can skip it**: If you have *paired* data (you know which x_0 corresponds to which x_1 and x_t), supervised MSE already enforces distributional matching. The GAN exists specifically because the reference application (biology) has *unpaired* marginals.

### Stage 3: CFM Velocity Field Training

Train a velocity field `v_theta(x_t, t)` to match the interpolant's velocity:

```
L_cfm = E[||v_theta(x_t, t) - dI/dt||^2]
```

where `x_t = I_t(x_0, x_1)` and `dI/dt` comes from differentiating the interpolant. The interpolant is frozen (detached from the graph).

At inference, integrate the velocity field with an ODE solver from t=0 to t=1.

## Technical Details

### Velocity Computation (dI/dt)

Analytically differentiating the interpolant:

```
dI/dt = (x_1 - x_0) + (1 - 2t)*C(x_0, x_1, t) + t*(1-t)*dC/dt
```

Three terms: linear velocity + correction value weighted by distance from midpoint + time derivative of the correction.

The reference implementation computes this using `torch.func.jacrev` + `vmap` over the batch. This works for small MLPs but is prohibitively expensive for large networks (UNets). Alternatives for high-dimensional data:
- **Forward-mode AD**: `torch.func.jvp` with tangent `dt=1` — one forward pass, exact
- **Finite differences**: `(I(t+h) - I(t-h)) / 2h` — two forward passes, approximate

### Time Noise Injection (t_smooth)

During training, the time input to the correction network is perturbed:

```
t_input = t + N(0, t_smooth^2)    # only during training
```

This regularizes the correction to be smooth w.r.t. time. Default `t_smooth = 0.01`.

### Regularization Options

The interpolant correction is regularized to prevent it from learning overly complex paths:

**Piecewise-linear**: Penalize deviation from a piecewise-linear path connecting observed marginals. For coupled samples `(x_0, x_t1, x_t2, ..., x_1)` at known times, construct the piecewise-linear interpolation between consecutive points and penalize `||I_t - piecewise_linear_t||^2`. This is the most commonly used regularizer in the reference.

**Linear**: Penalize `||I_t - linear_interp_t||^2` — i.e., the correction magnitude itself. Keeps paths close to straight lines between endpoints.

**2nd derivative (curvature)**: Minimize `integral ||d^2I/dt^2||^2 dt` estimated via finite differences: `(I(t+h) + I(t-h) - 2*I(t)) / h^2`. Encourages globally smooth paths.

**Landauer metric weighting**: Any of the above can be weighted by a diagonal metric `G` that is the inverse local variance of the data:

```
G(x, t) = 1 / (sum_i w_i * (x_i - x)^2 + rho)
w_i = exp(-||x - x_i||^2 / (2*gamma^2)) * exp(-(t - t_i)^2 / (2*t_gamma^2))
```

This downweights the penalty in dense regions (where data is plentiful) and upweights in sparse regions. The effect is to allow more correction where data is abundant and constrain paths more where there's little data to guide them.

### GAN Loss Variants

**Vanilla**:
```
L_D = softplus(-D(real)) + softplus(D(fake))
L_G = softplus(-D(fake))
```

**R3GAN (Relativistic with gradient penalties)**:
```
r1 = 0.5 * ||grad_x D(x_real)||^2
r2 = 0.5 * ||grad_x D(x_fake)||^2
L_D = softplus(D(fake) - D(real) + r1 + r2)
L_G = softplus(D(real) - D(fake))
```

The discriminator receives `concat([x_t, t])` — it is time-conditioned, since different times have different target distributions.

### Multi-Marginal OT Coupling (MMOT)

When marginals are unpaired, samples need to be coupled across time points. The reference uses a factorized approximation:

```
pi(x_0, x_t, x_1) ~ pi_1(x_0, x_t) * pi_2(x_t, x_1) / mu_t(x_t)
```

where `pi_1, pi_2` are pairwise OT plans computed independently. Sampling: fix x_t, sample x_0 from pi_1's column and x_1 from pi_2's row. This avoids computing the full K-marginal OT plan (which is intractable).

**Not needed when data is paired** (e.g., deterministic transformations of the same images).

### Alternative: Non-Parametric Interpolation (cubic_fm.py)

The reference also implements non-parametric multi-point interpolation (without the learned correction network):

- **Cubic spline** through all K observed points (scipy CubicSpline)
- **Lagrange polynomial** through all K points
- **Linear spline** (piecewise linear between consecutive points)

These define `mu_t(x)` — the mean of the probability path. Combined with an adaptive variance schedule `sigma_t` that vanishes at observed times:

```
x_t = mu_t + sigma_t * epsilon,    epsilon ~ N(0, I)
u_t(x|z) = (sigma'_t / sigma_t) * (x_t - mu_t) + mu'_t
```

The adaptive variance schedules (4 variants) all share the property of being zero at observed time points and maximal between them. Example (simplest):

```
sigma_t = M * sqrt(t_norm * (1 - t_norm))
```

where `t_norm` is t normalized to [0,1] within each interval between consecutive observations.

This approach doesn't need any adversarial training — it's fully defined by the observed data and the choice of interpolation/variance schedule. The learned interpolant approach (Stages 1-3 above) is more flexible but requires more training.

## Architecture (Reference Implementation)

All networks are shallow MLPs — appropriate for the low-dimensional biological data (2D-100D):

- **Interpolant correction**: `Linear(2*dim+1, h) -> ELU -> Linear(h, h) -> ELU -> Linear(h, dim)`
- **Discriminator**: `Linear(dim+1, h) -> ELU -> Linear(h, h) -> ELU -> Linear(h, 1)`
- **CFM velocity field**: `Linear(dim+1, h) -> SELU -> Linear(h, h) -> SELU -> Linear(h, h) -> SELU -> Linear(h, dim)`

Hidden dim h = 64 for low-dim, 1024 for 50D+.

## Future Direction

We are interested in applying this method to image data (e.g., CIFAR-10) where marginals correspond to deterministic degradations of images. Since degradation gives exact per-sample pairing, the GAN stage and OT coupling become unnecessary, simplifying the pipeline to supervised interpolant training + CFM velocity matching. The main adaptation needed is replacing MLPs with image-appropriate architectures (UNets with time conditioning).
