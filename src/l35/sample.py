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
  (lattice_x0_estimate/coord_x0_estimate) already validated in
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
    lattice_x0_estimate,
)
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
    max_timestep, generator=None,
):
    """Genuine consistency sampling (Song et al. 2023 Algorithm 1): one
    direct decoder call per NFE, each producing an x0 estimate straight
    from the boundary-condition formulas (lattice_x0_estimate /
    coord_x0_estimate), with the previous call's x0 estimate renoised to
    the next grid point before the next call -- no inner ODE/solver loop
    between calls (contrast few_step_sample, which runs num_steps small
    DDIM/PF-ODE solver steps). A student trained as a consistency function
    can be sampled this way at any num_steps without retraining; the
    boundary formulas being independent of num_steps is exactly what makes
    that possible.
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
            rand_l = torch.randn(l0_hat.shape, generator=generator).to(device)
            rand_x = torch.randn(x0_hat.shape, generator=generator).to(device)
            l_t = lattice_forward_noise(l0_hat, rand_l, ac_t)
            x_t = coord_forward_noise(x0_hat, rand_x, sigma_t_per_atom)

        with torch.no_grad():
            pred_eps, score = _decoder_step(
                decoder, l_t, x_t, atom_types, num_atoms, node2graph, t, sigma_t, sigma_norm_t
            )
            l0_hat = lattice_x0_estimate(l_t, pred_eps, ac_t)
            x0_hat = coord_x0_estimate(x_t, score, sigma_t_per_atom)

    return x0_hat, l0_hat
