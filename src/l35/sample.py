"""Deterministic few-step (DDIM lattice / VE-PFODE coordinate) sampler for
DiffCSP's CSP task: given real ground-truth composition (atom_types,
num_atoms), denoise frac_coords/lattice from noise in exactly num_steps
steps. Complements DiffCSP's own many-step CSPDiffusion.sample()
(diffusion.py), which is reused directly for the teacher side of an eval
rather than reimplemented here.

Noise initialization matches diffusion.py's sample() exactly: lattice ~
N(0, I) per graph, frac_coords ~ Uniform[0, 1) per atom -- same starting
distribution the teacher's own sampler uses, so a fair NFE-vs-quality
comparison starts both samplers from the same noise family.
"""
import torch

from src.l35.consistency_distill import coord_pfode_step, lattice_ddim_step
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
