"""L35 consistency-distillation math core for DiffCSP (dual-track: lattice +
fractional coordinates). Mirrors src/l37/consistency_distill.py's structure
(hardware-independent pure-math functions, verified against real teacher
source, not re-derived from memory).

DERIVATION (see docs/L35_PIPELINE_SPEC.md for full citations):

Lattice track (diffcsp/pl_modules/diffusion.py forward(): epsilon-prediction,
VP/DDPM). BetaScheduler zero-pads betas at index 0 (diff_utils.py:68), giving
alphas_cumprod[0] == 1 EXACTLY. The standard DDIM/Tweedie x0-estimate
  l0_hat = (l_t - sqrt(1-ac_t) * pred_eps) / sqrt(ac_t)
therefore satisfies the consistency boundary condition l0_hat(l_0, ac_t=1) ==
l_0 EXACTLY, for ANY pred_eps (its coefficient is exactly zero at the
boundary) -- no separate c_skip/c_out blend is needed; this formula already
IS a valid consistency function by construction. This is the DDIM ODE
(Song et al. 2020) applied to DiffCSP's own existing epsilon head -- the same
"reuse the model's existing structure" move as L37's GVP-coefficient reuse.

Coordinate track (VE-SDE / annealed Langevin on the wrapped torus; diffusion.py
sample()'s predictor step, citing Song & Ermon's score-SDE PC-sampler).
SigmaScheduler zero-pads sigmas at index 0 (diff_utils.py:97), giving
sigma_t=0 EXACTLY at the boundary. The Tweedie x0-estimate
  x0_hat = (x_t + sigma_t^2 * score) mod 1
is an approximation away from the boundary (Tweedie's formula is exact for
Gaussian noise; DiffCSP's forward process wraps mod 1, so the true kernel is
a WRAPPED normal, not Gaussian -- a real, flagged approximation, echoing the
L18/L11 wrapped-vs-Gaussian caveat elsewhere in this research thread).
AT the boundary specifically, sigma_t=0 kills the score term's coefficient
exactly regardless of that approximation, so x0_hat(x_0, sigma_t=0) == x_0
EXACTLY and unconditionally. The deterministic sampling step (for building
teacher trajectory pairs, not the boundary) is the VE probability-flow ODE
(Song et al. 2021, Eq. 13: dx = -0.5*g(t)^2*score*dt for zero-drift VE) --
exactly half of DiffCSP's own predictor step_size = sigma_t^2 - sigma_next^2,
with the stochastic term dropped.
"""
import torch

from src.l35.consistency_distill import (
    coord_pfode_step,
    coord_x0_estimate,
    lattice_ddim_step,
    lattice_x0_estimate,
    sample_index_pair,
)


class TestLatticeX0Estimate:
    def test_exact_boundary_at_full_alphas_cumprod_regardless_of_pred_eps(self):
        # ac_t=1 is DiffCSP's own scheduler value at t=0 (diff_utils.py: betas
        # zero-padded -> alphas_cumprod[0]==1 exactly). At this boundary the
        # formula must return l_t unchanged, no matter what pred_eps says.
        l_t = torch.randn(3, 3, 3)
        ac_t = torch.ones(3)

        for _ in range(3):
            garbage_pred_eps = torch.randn(3, 3, 3) * 100.0
            l0_hat = lattice_x0_estimate(l_t, garbage_pred_eps, ac_t)
            assert torch.allclose(l0_hat, l_t, atol=1e-6)

    def test_recovers_l0_when_pred_eps_is_the_true_noise(self):
        torch.manual_seed(0)
        l_0 = torch.randn(2, 3, 3)
        ac_t = torch.tensor([0.7, 0.3])
        true_eps = torch.randn(2, 3, 3)
        c0 = torch.sqrt(ac_t).view(-1, 1, 1)
        c1 = torch.sqrt(1 - ac_t).view(-1, 1, 1)
        l_t = c0 * l_0 + c1 * true_eps

        l0_hat = lattice_x0_estimate(l_t, true_eps, ac_t)

        assert torch.allclose(l0_hat, l_0, atol=1e-5)


class TestLatticeDdimStep:
    def test_stepping_to_ac_next_equal_to_ac_t_is_identity(self):
        # Zero-length step (ac_next == ac_t) must leave l_t unchanged --
        # sanity check on the DDIM step algebra before using it to build
        # nontrivial teacher-trajectory training pairs.
        torch.manual_seed(1)
        l_t = torch.randn(2, 3, 3)
        pred_eps = torch.randn(2, 3, 3)
        ac_t = torch.tensor([0.5, 0.5])

        l_next = lattice_ddim_step(l_t, pred_eps, ac_t, ac_t)

        assert torch.allclose(l_next, l_t, atol=1e-5)

    def test_stepping_all_the_way_to_ac_next_one_recovers_x0_estimate(self):
        # Stepping to ac_next=1 (the t=0 boundary) must land exactly on the
        # x0-estimate, since DDIM's step formula at zero target noise
        # collapses to the x0-estimate by construction.
        torch.manual_seed(2)
        l_t = torch.randn(2, 3, 3)
        pred_eps = torch.randn(2, 3, 3)
        ac_t = torch.tensor([0.4, 0.6])
        ac_next = torch.ones(2)

        l_next = lattice_ddim_step(l_t, pred_eps, ac_t, ac_next)
        l0_hat = lattice_x0_estimate(l_t, pred_eps, ac_t)

        assert torch.allclose(l_next, l0_hat, atol=1e-5)


class TestCoordX0Estimate:
    def test_exact_boundary_at_sigma_zero_regardless_of_score(self):
        # sigma_t=0 is DiffCSP's own scheduler value at t=0 (diff_utils.py:
        # sigmas zero-padded). At this boundary the sigma_t^2 prefactor kills
        # the score term exactly, so the formula must return x_t unchanged,
        # no matter what the score says -- this holds regardless of the
        # wrapped-vs-Gaussian Tweedie approximation error away from t=0.
        x_t = torch.rand(3, 5, 3)
        sigma_t = torch.zeros(3)

        for _ in range(3):
            garbage_score = torch.randn(3, 5, 3) * 100.0
            x0_hat = coord_x0_estimate(x_t, garbage_score, sigma_t)
            assert torch.allclose(x0_hat, x_t, atol=1e-6)

    def test_output_is_wrapped_into_unit_cell(self):
        torch.manual_seed(3)
        x_t = torch.rand(2, 5, 3)
        score = torch.randn(2, 5, 3) * 50.0  # large enough to push outside [0,1)
        sigma_t = torch.full((2,), 0.5)

        x0_hat = coord_x0_estimate(x_t, score, sigma_t)

        assert (x0_hat >= 0).all()
        assert (x0_hat < 1).all()


class TestCoordPfodeStep:
    def test_stepping_to_equal_sigma_is_identity(self):
        torch.manual_seed(4)
        x_t = torch.rand(2, 5, 3)
        score = torch.randn(2, 5, 3)
        sigma_t = torch.full((2,), 0.5)

        x_next = coord_pfode_step(x_t, score, sigma_t, sigma_t)

        assert torch.allclose(x_next, x_t % 1.0, atol=1e-6)

    def test_step_uses_half_of_diffcsp_own_predictor_coefficient(self):
        # DiffCSP's own predictor step_size = sigma_t**2 - sigma_next**2
        # (diffusion.py sample(), predictor block). The deterministic
        # PF-ODE substitute must use exactly HALF that coefficient with no
        # noise term (Song et al. 2021 Eq. 13 for zero-drift VE: drift has a
        # 1/2 factor vs the SDE's full g(t)^2 coefficient).
        torch.manual_seed(5)
        x_t = torch.rand(2, 5, 3)
        score = torch.randn(2, 5, 3)
        sigma_t = torch.tensor([0.8, 0.6])
        sigma_next = torch.tensor([0.5, 0.3])

        x_next = coord_pfode_step(x_t, score, sigma_t, sigma_next)

        expected_step_size = 0.5 * (sigma_t**2 - sigma_next**2)
        expected = (x_t - expected_step_size.view(-1, 1, 1) * score) % 1.0
        assert torch.allclose(x_next, expected, atol=1e-6)

    def test_output_is_wrapped_into_unit_cell(self):
        torch.manual_seed(6)
        x_t = torch.rand(2, 5, 3)
        score = torch.randn(2, 5, 3) * 50.0
        sigma_t = torch.full((2,), 0.8)
        sigma_next = torch.full((2,), 0.1)

        x_next = coord_pfode_step(x_t, score, sigma_t, sigma_next)

        assert (x_next >= 0).all()
        assert (x_next < 1).all()


class TestSampleIndexPair:
    def test_returns_adjacent_indices_on_uniform_integer_grid(self):
        idx_n, idx_next = sample_index_pair(num_steps=8, max_index=1000, batch_size=100)

        assert (idx_next > idx_n).all()
        # grid has num_steps+1 points spanning [0, max_index]; adjacent gap
        # should be a consistent ~max_index/num_steps
        gaps = (idx_next - idx_n).float()
        assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1.0)

    def test_indices_stay_within_bounds(self):
        idx_n, idx_next = sample_index_pair(num_steps=8, max_index=1000, batch_size=500)

        assert (idx_n >= 0).all()
        assert (idx_next <= 1000).all()

    def test_idx_next_never_hits_max_index_exactly(self):
        # max_index (e.g. 1000) is a real numerical cliff for lattice_x0_estimate:
        # DiffCSP's cosine schedule clips beta[max_index] to a 0.9999 ceiling,
        # crushing alphas_cumprod[max_index] two orders of magnitude below its
        # already-tiny neighbors (confirmed: ac[999]~=2.4e-6, ac[1000]~=2.4e-10),
        # so 1/sqrt(ac) explodes ~100x further right at that single index.
        # DiffCSP's own sample() never divides by sqrt(alphas_cumprod) at all
        # (it uses per-step alphas, not the cumulative product), so it never
        # hits this; our DDIM-style x0-estimate does, and needs the grid to
        # stay off this exact index. Large sample to make a miss vanishingly
        # unlikely if the exclusion isn't actually implemented.
        _, idx_next = sample_index_pair(num_steps=8, max_index=1000, batch_size=5000)

        assert (idx_next < 1000).all()

    def test_reproducible_with_generator(self):
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        idx_n1, idx_next1 = sample_index_pair(num_steps=8, max_index=1000, batch_size=10, generator=gen1)
        idx_n2, idx_next2 = sample_index_pair(num_steps=8, max_index=1000, batch_size=10, generator=gen2)

        assert torch.equal(idx_n1, idx_n2)
        assert torch.equal(idx_next1, idx_next2)
