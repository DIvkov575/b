"""Two few-step samplers for DiffCSP's CSP task, given real ground-truth
composition (atom_types, num_atoms):

- few_step_sample: deterministic DDIM (lattice) / VE-PFODE (coordinates)
  ODE solver, respaced onto a num_steps-point sub-grid of the full
  schedule. This is a step-count-compression technique (same idea DDIM
  itself introduced) -- it can run ANY decoder (teacher or a
  distillation-trained student) at ANY step count, but it does not exploit
  a consistency-distilled student's actual trained property: that its
  boundary-condition output already IS a direct x0 estimate from any
  noised input, not just an incremental denoising step.

- multistep_consistency_sample: genuine consistency sampling (Song et al.
  2023 Algorithm 1) -- one direct decoder call per NFE, each producing an
  x0 estimate straight from the boundary-condition formulas
  (lattice_x0_estimate_bounded/coord_x0_estimate) already validated in
  consistency_distill.py, with the PREVIOUS call's x0 estimate renoised to
  the next grid point before the next call (no inner ODE loop between
  calls). This is what lets ONE student, trained once against a
  reasonably fine training-time discretization, be sampled at 1, 4, 8, or
  16 steps at inference by varying only the grid -- the actual point of
  consistency distillation, which training one separate student per
  target NFE (this project's original approach) defeats.

Both complement DiffCSP's own many-step CSPDiffusion.sample() (diffusion.py),
which is reused directly for the teacher side of an eval at its native NFE
rather than reimplemented here.

Noise initialization matches diffusion.py's sample() exactly: lattice ~
N(0, I) per graph, frac_coords ~ Uniform[0, 1) per atom -- same starting
distribution the teacher's own sampler uses, so a fair NFE-vs-quality
comparison starts both samplers from the same noise family.
"""
import torch

from src.l35.consistency_distill import (
    coord_forward_noise,
    coord_pfode_step,
    coord_x0_estimate,
    lattice_ddim_step,
    lattice_forward_noise,
    lattice_x0_estimate_bounded,
)

# Empirical std of real MP-20 lattice matrix entries (n=100 structures via
# diffcsp's own process_one() + lattice_params_to_matrix_torch), the EDM-
# style sigma_data calibration lattice_x0_estimate_bounded needs. Measured,
# not assumed -- same convention EDM itself uses (sigma_data = std of the
# data the network is being asked to reconstruct).
LATTICE_SIGMA_DATA = 3.33
from src.l35.training_step import _decoder_step


def few_step_sample(
    decoder, atom_types, num_atoms, node2graph, num_steps, beta_scheduler, sigma_scheduler,
    max_timestep, generator=None,
):
    """Run exactly num_steps deterministic denoising steps from t=max_timestep
    down to t=0, using the same DDIM (lattice) / VE-PFODE (coordinates) math
    validated in consistency_distill.py and training_step.py.
    """
    device = num_atoms.device
    batch_size = num_atoms.shape[0]
    total_atoms = int(num_atoms.sum())

    l_t = torch.randn(batch_size, 3, 3, generator=generator, device=device)
    x_t = torch.rand(total_atoms, 3, generator=generator, device=device)

    # Grid excludes max_timestep itself (see sample_index_pair's docstring for
    # why: the cosine schedule's clipped beta[max_timestep] makes
    # alphas_cumprod there numerically pathological for a DDIM x0-estimate).
    grid = torch.linspace(0, max_timestep - 1, num_steps + 1).round().long().to(device)

    for i in range(num_steps, 0, -1):
        t_cur = grid[i].expand(batch_size)
        t_next = grid[i - 1].expand(batch_size)

        ac_cur = beta_scheduler.alphas_cumprod[t_cur]
        ac_next = beta_scheduler.alphas_cumprod[t_next]
        sigma_cur, sigma_norm_cur = sigma_scheduler.sigmas[t_cur], sigma_scheduler.sigmas_norm[t_cur]
        sigma_next = sigma_scheduler.sigmas[t_next]

        sigma_cur_per_atom = sigma_cur.repeat_interleave(num_atoms)
        sigma_next_per_atom = sigma_next.repeat_interleave(num_atoms)

        with torch.no_grad():
            pred_eps, score = _decoder_step(
                decoder, l_t, x_t, atom_types, num_atoms, node2graph, t_cur, sigma_cur, sigma_norm_cur
            )
            l_t = lattice_ddim_step(l_t, pred_eps, ac_cur, ac_next)
            x_t = coord_pfode_step(x_t, score, sigma_cur_per_atom, sigma_next_per_atom)

    return x_t, l_t


def consistency_sampling_grid(num_steps, max_timestep):
    """The sequence of timestep indices Algorithm 1 (Song et al. 2023) calls
    the consistency function at: tau_1=max_timestep-1 (max usable noise,
    excluding max_timestep itself for the same numerical-cliff reason
    documented in sample_index_pair), then descending to tau_{N-1}=0 for
    num_steps>1. For num_steps==1 this is the single one-shot-generation
    call at max noise -- there is no second point to renoise to, so the
    grid is the single-element [max_timestep-1], not [max_timestep-1, 0].

    Unlike few_step_sample's grid (num_steps+1 points, used as num_steps
    solver-step ENDPOINTS), this grid has exactly num_steps points, each a
    direct decoder-call TARGET -- the same grid works for any num_steps
    against the same trained student, which is the entire reason this
    function is separate from few_step_sample's grid construction.
    """
    if num_steps == 1:
        return torch.tensor([max_timestep - 1])
    return torch.linspace(max_timestep - 1, 0, num_steps).round().long()


def multistep_consistency_sample(
    decoder, atom_types, num_atoms, node2graph, num_steps, beta_scheduler, sigma_scheduler,
    max_timestep, generator=None, sigma_data_l=LATTICE_SIGMA_DATA,
):
    """Genuine consistency sampling (Song et al. 2023 Algorithm 1): one
    direct decoder call per NFE, each producing an x0 estimate straight
    from the boundary-condition formulas, with the previous call's x0
    estimate renoised to the next grid point before the next call -- no
    inner ODE/solver loop between calls (contrast few_step_sample, which
    runs num_steps small DDIM/PF-ODE solver steps). A student trained as a
    consistency function can be sampled this way at any num_steps without
    retraining; the boundary formulas being independent of num_steps is
    exactly what makes that possible.

    Uses lattice_x0_estimate_bounded (the EDM/Consistency-Models-style
    c_skip/c_out blend), NOT the naive lattice_x0_estimate -- the naive
    formula's pred_eps coefficient sqrt(1-ac_t)/sqrt(ac_t) is unbounded as
    ac_t -> 0 (~642x at DiffCSP's real ac[999]=2.428e-6), confirmed to
    produce geometrically invalid structures (collapsed/negative-
    determinant lattices) regardless of network quality -- an untrained
    teacher fails identically to a trained student under the naive
    formula, and a direct overfit test shows the consistency-distillation
    loss doesn't even converge when regressing against ground truth at
    that ac_t. See consistency_distill.py's lattice_c_skip/lattice_c_out
    docstrings for the full derivation.

    coord_x0_estimate (unchanged) has its OWN, separate, and still-open
    issue: its coefficient (sigma_t**2, bounded, unlike the lattice's) is
    NOT the problem -- confirmed directly, it stays under ~0.2 across the
    whole schedule -- but the network's raw output on renoised, off-
    manifold inputs (never seen in training, which only ever regresses
    against real ground truth) grows far outside its real in-distribution
    range (measured: real scale spans ~0.001 at t=999 to ~475 at t=1; a
    naive fixed clamp tested during debugging discarded the network's
    actual output at 100% of steps past the first two, a degenerate non-
    fix, not included here). Coordinate collapse under this sampler is
    real and NOT resolved by the lattice fix alone.
    """
    device = num_atoms.device
    batch_size = num_atoms.shape[0]
    total_atoms = int(num_atoms.sum())

    grid = consistency_sampling_grid(num_steps, max_timestep).to(device)

    l_t = torch.randn(batch_size, 3, 3, generator=generator, device=device)
    x_t = torch.rand(total_atoms, 3, generator=generator, device=device)

    l0_hat, x0_hat = None, None
    for i, t_scalar in enumerate(grid):
        t = t_scalar.expand(batch_size)
        ac_t = beta_scheduler.alphas_cumprod[t]
        sigma_t, sigma_norm_t = sigma_scheduler.sigmas[t], sigma_scheduler.sigmas_norm[t]
        sigma_t_per_atom = sigma_t.repeat_interleave(num_atoms)

        if i > 0:
            # Renoise the previous call's x0 estimate up to this grid
            # point's noise level (Algorithm 1's "x^tau_n <- x + noise"),
            # not the raw previous NOISED input -- consistency sampling
            # walks between x0 ESTIMATES, never between intermediate
            # noised states the way an ODE solver does.
            rand_l = torch.randn(l0_hat.shape, generator=generator, device=device)
            rand_x = torch.randn(x0_hat.shape, generator=generator, device=device)
            l_t = lattice_forward_noise(l0_hat, rand_l, ac_t)
            x_t = coord_forward_noise(x0_hat, rand_x, sigma_t_per_atom)

        with torch.no_grad():
            pred_eps, score = _decoder_step(
                decoder, l_t, x_t, atom_types, num_atoms, node2graph, t, sigma_t, sigma_norm_t
            )
            l0_hat = lattice_x0_estimate_bounded(l_t, pred_eps, ac_t, sigma_data_l)
            x0_hat = coord_x0_estimate(x_t, score, sigma_t_per_atom)

    return x0_hat, l0_hat
