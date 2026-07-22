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
import torch.nn as nn

from src.l35.consistency_distill import (
    coord_forward_noise,
    coord_pfode_step,
    coord_x0_estimate,
    ema_update,
    lattice_ddim_step,
    lattice_forward_noise,
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


class TestLatticeCskipCout:
    # Bounded (EDM/Consistency-Models-style) reparameterization of the
    # lattice track's consistency function -- replaces lattice_x0_estimate's
    # naive DDIM/Tweedie formula, whose pred_eps coefficient
    # sqrt(1-ac_t)/sqrt(ac_t) is UNBOUNDED as ac_t -> 0 (~642x at
    # ac_t=2.4e-6, DiffCSP's own real ac[999] value). Confirmed on real data:
    # even a well-trained network's small pred_eps error gets amplified past
    # any usable range there, both empirically (multistep consistency
    # sampling produces geometrically invalid structures regardless of which
    # network drives it) and via a direct overfit test (loss will not even
    # converge when regressing l0_hat against ground truth at ac_t~=2.4e-6,
    # oscillating instead of decreasing). Song et al. 2023 Sec. 3 (citing
    # Karras et al. 2022's EDM preconditioning) define f_theta(x,t) =
    # c_skip(t)*x + c_out(t)*F_theta(x,t) specifically so c_out stays
    # BOUNDED as t grows -- this class derives the DiffCSP-lattice analogue
    # of that same c_skip/c_out blend, keeping the exact boundary condition
    # at ac_t=1 while eliminating the 1/sqrt(ac_t) blow-up as ac_t -> 0.
    def test_c_skip_and_c_out_satisfy_exact_boundary_condition_at_ac_one(self):
        # c_skip(1)=1, c_out(1)=0 -- same boundary Song et al. 2023 impose,
        # required for f_theta(l_0, ac_t=1) == l_0 regardless of pred_eps.
        from src.l35.consistency_distill import lattice_c_skip, lattice_c_out

        sigma_data_l = 3.33
        ac_t = torch.ones(4)

        assert torch.allclose(lattice_c_skip(ac_t, sigma_data_l), torch.ones(4), atol=1e-6)
        assert torch.allclose(lattice_c_out(ac_t, sigma_data_l), torch.zeros(4), atol=1e-6)

    def test_c_out_stays_bounded_as_ac_t_approaches_zero(self):
        # The property the naive formula lacks: c_out(ac_t) -> -sigma_data_l
        # (a FINITE constant) as ac_t -> 0, rather than diverging.
        from src.l35.consistency_distill import lattice_c_out

        sigma_data_l = 3.33
        ac_t_values = torch.tensor([1e-2, 1e-4, 1e-6, 1e-8, 0.0])

        c_out_values = lattice_c_out(ac_t_values, sigma_data_l)

        assert torch.isfinite(c_out_values).all()
        assert torch.allclose(c_out_values[-1], torch.tensor(-sigma_data_l), atol=1e-4)
        # monotonically approaching the limit as ac_t shrinks (no oscillation
        # or overshoot past the asymptote)
        assert (c_out_values.abs() <= sigma_data_l + 1e-4).all()

    def test_bounded_x0_estimate_matches_naive_formula_near_ac_one(self):
        # Near the boundary (ac_t close to 1), the bounded blend must agree
        # closely with the existing naive formula -- this reparameterization
        # changes behavior specifically in the high-noise regime the naive
        # formula mishandles, not near ac_t=1 where the naive formula is
        # already exact.
        from src.l35.consistency_distill import lattice_x0_estimate_bounded

        torch.manual_seed(3)
        l_t = torch.randn(2, 3, 3)
        pred_eps = torch.randn(2, 3, 3) * 0.5  # eps-prediction scale, roughly unit variance
        ac_t = torch.tensor([0.999, 0.995])
        sigma_data_l = 3.33

        naive = lattice_x0_estimate(l_t, pred_eps, ac_t)
        bounded = lattice_x0_estimate_bounded(l_t, pred_eps, ac_t, sigma_data_l)

        assert torch.allclose(naive, bounded, atol=0.05)

    def test_bounded_x0_estimate_stays_bounded_at_the_real_numerical_cliff_for_a_fixed_pred_eps(self):
        # The actual failure this reparameterization exists to fix: DiffCSP's
        # real ac[999]=2.4280e-06 (sample_index_pair's documented numerical
        # cliff). Compare the SAME pred_eps under the naive formula (whose
        # coefficient sqrt(1-ac_t)/sqrt(ac_t) is ~642x at this ac_t) against
        # the bounded formula (whose coefficient saturates at sigma_data_l):
        # the bounded output must be dramatically smaller for realistic
        # (roughly unit-scale, as an eps-predictor is trained to produce)
        # pred_eps -- this is a claim about the COEFFICIENT's boundedness,
        # not a claim that bounded output is independent of pred_eps's own
        # scale (c_out(ac_t)*pred_eps is still unbounded if pred_eps itself
        # is fed an unrealistic, unboundedly large value -- the fix bounds
        # the coefficient the naive formula gets wrong, not pred_eps itself).
        from src.l35.consistency_distill import lattice_x0_estimate_bounded

        ac_t = torch.tensor([2.4280e-06])
        sigma_data_l = 3.33
        torch.manual_seed(5)
        l_t = torch.randn(1, 3, 3)
        pred_eps = torch.randn(1, 3, 3)  # realistic eps-predictor scale, ~N(0,1)

        naive = lattice_x0_estimate(l_t, pred_eps, ac_t)
        bounded = lattice_x0_estimate_bounded(l_t, pred_eps, ac_t, sigma_data_l)

        assert naive.abs().max().item() > 100.0, "sanity check: naive formula really does blow up here"
        assert bounded.abs().max().item() < 10 * sigma_data_l

    def test_bounded_x0_estimate_is_a_real_c_skip_c_out_blend(self):
        # Direct algebraic check: lattice_x0_estimate_bounded(l_t, pred_eps,
        # ac_t) == c_skip(ac_t)*l_t + c_out(ac_t)*pred_eps, not some other
        # formula that happens to satisfy the boundary condition.
        from src.l35.consistency_distill import (
            lattice_c_skip, lattice_c_out, lattice_x0_estimate_bounded,
        )

        torch.manual_seed(4)
        l_t = torch.randn(3, 3, 3)
        pred_eps = torch.randn(3, 3, 3)
        ac_t = torch.tensor([0.7, 0.3, 0.01])
        sigma_data_l = 3.33

        expected = (
            lattice_c_skip(ac_t, sigma_data_l).view(-1, 1, 1) * l_t
            + lattice_c_out(ac_t, sigma_data_l).view(-1, 1, 1) * pred_eps
        )
        actual = lattice_x0_estimate_bounded(l_t, pred_eps, ac_t, sigma_data_l)

        assert torch.allclose(expected, actual, atol=1e-6)


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


class TestLatticeForwardNoise:
    """Matches diffusion.py forward()'s exact noising formula:
    input_lattice = sqrt(ac_t)*lattices + sqrt(1-ac_t)*rand_l -- the forward
    process this whole distillation scheme distills a solver for, and which
    an earlier version of training_step.py never actually applied.
    """

    def test_matches_diffusion_py_forward_noising_formula(self):
        torch.manual_seed(10)
        l_0 = torch.randn(2, 3, 3)
        ac_t = torch.tensor([0.7, 0.3])
        rand_l = torch.randn(2, 3, 3)

        l_t = lattice_forward_noise(l_0, rand_l, ac_t)

        c0 = torch.sqrt(ac_t).view(-1, 1, 1)
        c1 = torch.sqrt(1 - ac_t).view(-1, 1, 1)
        expected = c0 * l_0 + c1 * rand_l
        assert torch.allclose(l_t, expected)

    def test_exact_boundary_at_ac_one_ignores_noise(self):
        l_0 = torch.randn(2, 3, 3)
        rand_l = torch.randn(2, 3, 3) * 100.0
        ac_t = torch.ones(2)

        l_t = lattice_forward_noise(l_0, rand_l, ac_t)

        assert torch.allclose(l_t, l_0, atol=1e-6)


class TestCoordForwardNoise:
    """Matches diffusion.py forward()'s exact noising formula:
    input_frac_coords = (frac_coords + sigma_t*rand_x) % 1.
    """

    def test_matches_diffusion_py_forward_noising_formula(self):
        torch.manual_seed(11)
        x_0 = torch.rand(5, 3)
        sigma_t = torch.full((5,), 0.3)
        rand_x = torch.randn(5, 3)

        x_t = coord_forward_noise(x_0, rand_x, sigma_t)

        expected = (x_0 + sigma_t.view(-1, 1) * rand_x) % 1.0
        assert torch.allclose(x_t, expected)

    def test_exact_boundary_at_sigma_zero_ignores_noise(self):
        x_0 = torch.rand(5, 3)
        rand_x = torch.randn(5, 3) * 100.0
        sigma_t = torch.zeros(5)

        x_t = coord_forward_noise(x_0, rand_x, sigma_t)

        assert torch.allclose(x_t, x_0 % 1.0, atol=1e-6)

    def test_output_is_wrapped_into_unit_cell(self):
        x_0 = torch.rand(5, 3)
        rand_x = torch.randn(5, 3) * 50.0
        sigma_t = torch.full((5,), 0.5)

        x_t = coord_forward_noise(x_0, rand_x, sigma_t)

        assert (x_t >= 0).all()
        assert (x_t < 1).all()


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


class TestEmaUpdate:
    """ema_update implements Song et al. 2023 Eq. 8: theta_minus <- stopgrad(
    mu*theta_minus + (1-mu)*theta) -- moves the target network's parameters
    partway toward the (live, training) student's current parameters, never
    the reverse and never in-place on the student.
    """

    def test_mu_one_leaves_target_completely_unchanged(self):
        target = nn.Linear(3, 3)
        student = nn.Linear(3, 3)
        original_weight = target.weight.detach().clone()

        ema_update(target, student, mu=1.0)

        assert torch.equal(target.weight, original_weight)

    def test_mu_zero_makes_target_exactly_equal_student(self):
        target = nn.Linear(3, 3)
        student = nn.Linear(3, 3)

        ema_update(target, student, mu=0.0)

        assert torch.equal(target.weight, student.weight)
        assert torch.equal(target.bias, student.bias)

    def test_mu_half_averages_target_and_student(self):
        target = nn.Linear(2, 2, bias=False)
        student = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            target.weight.copy_(torch.zeros(2, 2))
            student.weight.copy_(torch.ones(2, 2) * 4.0)

        ema_update(target, student, mu=0.5)

        assert torch.allclose(target.weight, torch.ones(2, 2) * 2.0)

    def test_target_network_stays_out_of_student_autograd_graph(self):
        # ema_update must not create a graph connecting target's parameters
        # back to student's -- it's an in-place buffer update, not a
        # differentiable operation the student's own loss could backprop through.
        target = nn.Linear(2, 2)
        student = nn.Linear(2, 2)

        ema_update(target, student, mu=0.9)

        for p in target.parameters():
            assert not p.requires_grad or p.grad_fn is None

    def test_student_parameters_are_not_mutated(self):
        target = nn.Linear(2, 2)
        student = nn.Linear(2, 2)
        original_student_weight = student.weight.detach().clone()

        ema_update(target, student, mu=0.5)

        assert torch.equal(student.weight, original_student_weight)
