"""Tests for the L37 consistency-distillation math core (src/l37/consistency_distill.py).

Ground truth for the boundary-condition parametrization and the GVP path comes from
MDGen's own `mdgen/transport/path.py` (GVPCPlan): alpha_t = sin(t*pi/2), sigma_t =
cos(t*pi/2), with t=0 the noise endpoint and t=1 the data endpoint. These tests do not
import MDGen itself (kept hardware/checkpoint independent) but the constants below are
transcribed from that file so the parametrization is provably consistent with it.
"""
import math

import torch

from src.l37.consistency_distill import (
    gvp_alpha_sigma,
    consistency_output,
    euler_step,
    sample_timestep_pair,
    masked_mse,
)


def _alpha_sigma_reference(t):
    return math.sin(t * math.pi / 2), math.cos(t * math.pi / 2)


def test_gvp_alpha_sigma_matches_mdgen_path_at_boundaries():
    t = torch.tensor([0.0, 1.0])
    alpha, sigma = gvp_alpha_sigma(t)
    assert torch.allclose(alpha, torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(sigma, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_gvp_alpha_sigma_matches_reference_at_midpoint():
    t = torch.tensor([0.3])
    alpha, sigma = gvp_alpha_sigma(t)
    ref_alpha, ref_sigma = _alpha_sigma_reference(0.3)
    assert torch.allclose(alpha, torch.tensor([ref_alpha]), atol=1e-6)
    assert torch.allclose(sigma, torch.tensor([ref_sigma]), atol=1e-6)


def test_consistency_output_at_data_endpoint_equals_x_regardless_of_raw_output():
    # boundary condition: f_theta(x, t=1) == x exactly, for ANY raw network output.
    x = torch.randn(2, 3, 4)
    raw_output = torch.randn(2, 3, 4) * 100  # deliberately large/arbitrary
    t = torch.ones(2)
    out = consistency_output(raw_output, x, t)
    assert torch.allclose(out, x, atol=1e-5)


def test_consistency_output_at_noise_endpoint_equals_raw_output():
    # at t=0: c_skip=alpha(0)=0, c_out=sigma(0)=1, so f_theta(x,0) == raw_output.
    x = torch.randn(2, 3, 4)
    raw_output = torch.randn(2, 3, 4)
    t = torch.zeros(2)
    out = consistency_output(raw_output, x, t)
    assert torch.allclose(out, raw_output, atol=1e-5)


def test_consistency_output_general_t_matches_closed_form():
    x = torch.randn(2, 3, 4)
    raw_output = torch.randn(2, 3, 4)
    t = torch.full((2,), 0.4)
    alpha, sigma = gvp_alpha_sigma(t)
    expected = alpha.view(2, 1, 1) * x + sigma.view(2, 1, 1) * raw_output
    out = consistency_output(raw_output, x, t)
    assert torch.allclose(out, expected, atol=1e-5)


def test_euler_step_linear_extrapolation():
    x = torch.tensor([[1.0, 2.0]])
    velocity = torch.tensor([[3.0, -1.0]])
    t_n = torch.tensor([0.2])
    t_next = torch.tensor([0.5])
    out = euler_step(x, t_n, t_next, velocity)
    # x + (t_next - t_n) * velocity = [1,2] + 0.3*[3,-1] = [1.9, 1.7]
    assert torch.allclose(out, torch.tensor([[1.9, 1.7]]), atol=1e-6)


def test_euler_step_zero_dt_is_identity():
    x = torch.randn(2, 3)
    velocity = torch.randn(2, 3)
    t = torch.full((2,), 0.5)
    out = euler_step(x, t, t, velocity)
    assert torch.allclose(out, x, atol=1e-6)


def test_sample_timestep_pair_shapes_and_ordering():
    gen = torch.Generator().manual_seed(0)
    t_n, t_next = sample_timestep_pair(num_steps=4, batch_size=8, generator=gen)
    assert t_n.shape == (8,)
    assert t_next.shape == (8,)
    assert torch.all(t_next > t_n)
    assert torch.all(t_n >= 0.0) and torch.all(t_n < 1.0)
    assert torch.all(t_next <= 1.0)


def test_sample_timestep_pair_lands_on_grid():
    # with num_steps=4, the grid is {0, 0.25, 0.5, 0.75, 1.0}; every sampled t_n, t_next
    # must be one of these five values, and t_next must be exactly one grid step ahead.
    gen = torch.Generator().manual_seed(1)
    t_n, t_next = sample_timestep_pair(num_steps=4, batch_size=32, generator=gen)
    grid = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    for a, b in zip(t_n, t_next):
        assert torch.any(torch.isclose(grid, a, atol=1e-6))
        assert torch.any(torch.isclose(grid, b, atol=1e-6))
        assert torch.isclose(b - a, torch.tensor(0.25), atol=1e-6)


def test_sample_timestep_pair_is_deterministic_given_generator_seed():
    gen1 = torch.Generator().manual_seed(42)
    gen2 = torch.Generator().manual_seed(42)
    t_n1, t_next1 = sample_timestep_pair(num_steps=8, batch_size=16, generator=gen1)
    t_n2, t_next2 = sample_timestep_pair(num_steps=8, batch_size=16, generator=gen2)
    assert torch.equal(t_n1, t_n2)
    assert torch.equal(t_next1, t_next2)


def test_masked_mse_matches_plain_mse_when_mask_is_all_ones():
    a = torch.randn(4, 5, 3)
    b = torch.randn(4, 5, 3)
    mask = torch.ones(4, 5, 3)
    got = masked_mse(a, b, mask)
    expected = ((a - b) ** 2).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_masked_mse_ignores_masked_positions():
    a = torch.zeros(1, 4)
    b = torch.zeros(1, 4)
    b[0, 3] = 100.0  # a huge error at one position
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])  # mask out that position
    got = masked_mse(a, b, mask)
    assert torch.allclose(got, torch.tensor(0.0), atol=1e-6)


def test_masked_mse_per_batch_then_averaged():
    # two batch elements, each with its own mask denominator (mirrors MDGen's mean_flat,
    # which normalizes per-example before averaging over the batch).
    a = torch.zeros(2, 4)
    b = torch.zeros(2, 4)
    b[0, 0] = 2.0  # error 4, masked-mean over 1 valid position for row 0 = 4
    b[1, :] = 1.0  # error 1 each, masked-mean over 4 valid positions for row 1 = 1
    mask = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    got = masked_mse(a, b, mask)
    # row 0: mean((2-0)^2 over 1 unmasked elt) = 4.0; row 1: mean(1 over 4 elts) = 1.0
    # batch mean = (4.0 + 1.0) / 2 = 2.5
    assert torch.allclose(got, torch.tensor(2.5), atol=1e-6)
