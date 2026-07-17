"""Consistency-distillation math core for L37 (MDGen few-step distillation, method-only).

Boundary-condition parametrization reuses MDGen's own GVP path coefficients (verified
against `mdgen/transport/path.py::GVPCPlan`, not re-derived): alpha_t = sin(t*pi/2) is
the data-endpoint coefficient, sigma_t = cos(t*pi/2) is the noise-endpoint coefficient,
with t=0 the noise endpoint and t=1 the data endpoint. Using (alpha_t, sigma_t) as
(c_skip, c_out) gives f_theta(x, 1) = x exactly, for any raw network output -- the
consistency function's boundary condition -- with no new hyperparameters.
"""
import math

import torch


def gvp_alpha_sigma(t):
    """GVP path coefficients (alpha_t, sigma_t), matching mdgen/transport/path.py."""
    alpha_t = torch.sin(t * math.pi / 2)
    sigma_t = torch.cos(t * math.pi / 2)
    return alpha_t, sigma_t


def consistency_output(raw_output, x, t):
    """f_theta(x, t) = alpha_t * x + sigma_t * raw_output (boundary condition at t=1)."""
    alpha_t, sigma_t = gvp_alpha_sigma(t)
    shape = (t.shape[0],) + (1,) * (x.dim() - 1)
    return alpha_t.view(shape) * x + sigma_t.view(shape) * raw_output


def euler_step(x, t_n, t_next, velocity):
    """One explicit Euler step of the flow ODE from t_n to t_next."""
    dt = (t_next - t_n).view((t_n.shape[0],) + (1,) * (x.dim() - 1))
    return x + dt * velocity


def sample_timestep_pair(num_steps, batch_size, generator=None):
    """Sample adjacent timestep pairs (t_n, t_next) on the {0, 1/N, ..., 1} grid."""
    grid = torch.linspace(0.0, 1.0, num_steps + 1)
    idx = torch.randint(0, num_steps, (batch_size,), generator=generator)
    return grid[idx], grid[idx + 1]


def masked_mse(a, b, mask):
    """Mean squared error, normalized per-example by mask sum then averaged over batch.

    Mirrors MDGen's own mean_flat: normalize each example by its own valid-element
    count before averaging over the batch, so examples with different mask sizes
    contribute equally rather than being dominated by whichever has more valid elements.
    """
    sq_err = (a - b) ** 2 * mask
    dims = list(range(1, a.dim()))
    per_example = sq_err.sum(dim=dims) / mask.sum(dim=dims)
    return per_example.mean()
