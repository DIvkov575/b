"""Consistency-distillation math core for L35 (DiffCSP few-step distillation,
dual-track: lattice + fractional coordinates). See tests/l35/test_consistency_distill.py
for the full derivation and citations (DDIM boundary condition on the lattice's
epsilon-prediction VP track; VE probability-flow ODE, Song et al. 2021 Eq. 13,
on the coordinate track's wrapped-normal score). Both boundary formulas are
verified exact at DiffCSP's own scheduler zero-padding (alphas_cumprod[0]==1,
sigmas[0]==0); the coordinate x0-estimate away from that boundary is Tweedie's
formula applied to a wrapped (not Gaussian) kernel -- a flagged approximation.
"""
import torch


def lattice_x0_estimate(l_t, pred_eps, alphas_cumprod_t):
    """DDIM/Tweedie x0-estimate for the lattice's VP/epsilon-prediction track.

    Exact at alphas_cumprod_t=1 (DiffCSP's own t=0 scheduler value) for any
    pred_eps, since the pred_eps coefficient sqrt(1-ac_t) is exactly zero there.
    """
    shape = (l_t.shape[0],) + (1,) * (l_t.dim() - 1)
    c0 = torch.sqrt(alphas_cumprod_t).view(shape)
    c1 = torch.sqrt(1.0 - alphas_cumprod_t).view(shape)
    return (l_t - c1 * pred_eps) / c0


def lattice_ddim_step(l_t, pred_eps, alphas_cumprod_t, alphas_cumprod_next):
    """Deterministic DDIM step from ac_t to ac_next, reusing the same pred_eps
    (constant-noise-direction assumption standard to DDIM, Song et al. 2020 Eq. 12).
    """
    shape = (l_t.shape[0],) + (1,) * (l_t.dim() - 1)
    l0_hat = lattice_x0_estimate(l_t, pred_eps, alphas_cumprod_t)
    c0_next = torch.sqrt(alphas_cumprod_next).view(shape)
    c1_next = torch.sqrt(1.0 - alphas_cumprod_next).view(shape)
    return c0_next * l0_hat + c1_next * pred_eps


def coord_x0_estimate(x_t, score, sigma_t):
    """Tweedie x0-estimate for the coordinate track's VE score, wrapped to [0,1).

    Exact at sigma_t=0 (DiffCSP's own t=0 scheduler value) for any score,
    since the score coefficient sigma_t**2 is exactly zero there. Away from
    the boundary this is an approximation: Tweedie's formula is exact for a
    Gaussian transition kernel, but DiffCSP's forward process wraps mod 1,
    making the true kernel a wrapped normal, not Gaussian.
    """
    shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
    sigma_sq = (sigma_t**2).view(shape)
    return (x_t + sigma_sq * score) % 1.0


def coord_pfode_step(x_t, score, sigma_t, sigma_next):
    """Deterministic VE probability-flow-ODE step (Song et al. 2021 Eq. 13:
    dx = -0.5*g(t)^2*score*dt for zero-drift VE), discretized with DiffCSP's
    own step_size = sigma_t**2 - sigma_next**2 (diffusion.py predictor block),
    halved per Eq. 13, with the stochastic term dropped for determinism.
    """
    shape = (x_t.shape[0],) + (1,) * (x_t.dim() - 1)
    step_size = 0.5 * (sigma_t**2 - sigma_next**2).view(shape)
    return (x_t - step_size * score) % 1.0


def sample_index_pair(num_steps, max_index, batch_size, generator=None):
    """Sample adjacent index pairs on a grid over [0, max_index - 1], for
    consistency-training pairs.

    Excludes max_index itself from ever being an endpoint: DiffCSP's cosine
    beta schedule clips beta[max_index] to a 0.9999 ceiling, crushing
    alphas_cumprod[max_index] roughly two orders of magnitude below its
    already-tiny neighbors (e.g. ac[999]~=2.4e-6 vs ac[1000]~=2.4e-10 at
    timesteps=1000) -- lattice_x0_estimate's 1/sqrt(ac) blows up there.
    DiffCSP's own sample() never hits this (it divides by per-step alphas,
    not the cumulative product), so max_index is a real numerical cliff
    specific to this consistency-distillation formulation, not a limitation
    inherited from the teacher's own math.
    """
    grid = torch.linspace(0, max_index - 1, num_steps + 1).round().long()
    idx = torch.randint(0, num_steps, (batch_size,), generator=generator)
    return grid[idx], grid[idx + 1]
