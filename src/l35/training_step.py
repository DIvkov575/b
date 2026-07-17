"""L35 consistency-distillation training step for DiffCSP's dual-track (lattice +
fractional-coordinate) denoiser.

Composes src/l35/consistency_distill.py's boundary-condition math with a real
decoder's calling convention. Two teacher-output conventions are load-bearing
and were taken directly from diffcsp/pl_modules/diffusion.py, not re-derived:
  - pred_l IS the epsilon prediction directly (forward(): loss_lattice against rand_l).
  - pred_x is a NORMALIZED score; sample() rescales it via `pred_x * sqrt(sigma_norm)`
    before using it as the true score fed to the annealed-Langevin update.

Self-distillation (student targets itself one step ahead), matching L37's
consistency_distill training_step structure -- teacher is frozen/stop-gradient,
student is trained so its own boundary-condition output at t_n matches a
stop-gradient copy of its output at t_next, where t_next's input was produced
by ONE deterministic teacher step from t_n.
"""
import torch

from src.l35.consistency_distill import (
    coord_pfode_step,
    coord_x0_estimate,
    lattice_ddim_step,
    lattice_x0_estimate,
    sample_index_pair,
)


def _sinusoidal_time_embedding(times, dim=256):
    """Matches diffcsp.pl_modules.diffusion.SinusoidalTimeEmbeddings exactly
    (same half_dim log-spaced frequency construction) -- decoders were trained
    against embeddings from that exact class, not a generic sinusoidal scheme.
    """
    half_dim = dim // 2
    emb_scale = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, device=times.device) * -emb_scale)
    args = times[:, None].float() * freqs[None, :]
    return torch.cat([args.sin(), args.cos()], dim=-1)


def _decoder_step(decoder, l_t, x_t, atom_types, num_atoms, node2graph, t_index, sigma_t, sigma_norm_t):
    """One decoder call producing (pred_eps, true_score), applying the real
    un-normalization convention (`pred_x * sqrt(sigma_norm)`) from sample().
    """
    time_emb = _sinusoidal_time_embedding(t_index.float())
    pred_l, pred_x = decoder(time_emb, atom_types, x_t, l_t, num_atoms, node2graph)
    sigma_norm_per_atom = sigma_norm_t.repeat_interleave(num_atoms).view(-1, 1)
    true_score = pred_x * torch.sqrt(sigma_norm_per_atom)
    return pred_l, true_score


def consistency_distillation_loss(
    teacher, student, batch, num_steps, beta_scheduler, sigma_scheduler, generator=None
):
    """One consistency-distillation training step over a toy/real DiffCSP batch.

    Args:
        teacher: frozen decoder, callable as CSPNet (time_emb, atom_types, frac_coords,
            lattices, num_atoms, node2graph) -> (pred_l, pred_x). No gradient flows into
            it regardless of its own requires_grad state (wrapped in no_grad).
        student: trainable decoder with the same calling convention.
        batch: dict with keys num_atoms, node2graph, atom_types, frac_coords, lattices.
        num_steps: number of steps on DiffCSP's {0, ..., timesteps} grid to sample from.
        beta_scheduler: a real diffcsp.pl_modules.diff_utils.BetaScheduler instance
            (or matching interface: .timesteps, .alphas_cumprod). sigma_norm is a
            10,000-sample Monte Carlo estimate (diff_utils.sigma_norm) with no closed
            form -- callers MUST supply the real scheduler, not a re-derived approximation.
        sigma_scheduler: a real diffcsp.pl_modules.diff_utils.SigmaScheduler instance
            (or matching interface: .timesteps, .sigmas, .sigmas_norm).
        generator: optional torch.Generator for reproducible timestep-pair sampling.

    Returns:
        (loss, aux) where aux has the sampled 't_n'/'t_next' index tensors (per-graph).
    """
    num_atoms = batch["num_atoms"]
    node2graph = batch["node2graph"]
    atom_types = batch["atom_types"]
    batch_size = num_atoms.shape[0]
    max_timestep = beta_scheduler.timesteps

    idx_n, idx_next = sample_index_pair(num_steps, max_timestep, batch_size, generator=generator)
    idx_n = idx_n.to(num_atoms.device)
    idx_next = idx_next.to(num_atoms.device)

    ac_n = beta_scheduler.alphas_cumprod[idx_n]
    ac_next = beta_scheduler.alphas_cumprod[idx_next]
    sigma_n, sigma_norm_n = sigma_scheduler.sigmas[idx_n], sigma_scheduler.sigmas_norm[idx_n]
    sigma_next = sigma_scheduler.sigmas[idx_next]
    sigma_norm_next = sigma_scheduler.sigmas_norm[idx_next]

    # coord_*/x_n/score are per-ATOM (num_nodes, 3); sigma_n/sigma_next are
    # per-GRAPH (batch_size,) -- same repeat_interleave(num_atoms) pattern
    # diffusion.py's forward() uses for sigmas_per_atom.
    sigma_n_per_atom = sigma_n.repeat_interleave(num_atoms)
    sigma_next_per_atom = sigma_next.repeat_interleave(num_atoms)

    l_n = batch["lattices"]
    x_n = batch["frac_coords"]

    with torch.no_grad():
        pred_l_n, score_n = _decoder_step(
            teacher, l_n, x_n, atom_types, num_atoms, node2graph, idx_n, sigma_n, sigma_norm_n
        )
        l_next = lattice_ddim_step(l_n, pred_l_n, ac_n, ac_next)
        x_next = coord_pfode_step(x_n, score_n, sigma_n_per_atom, sigma_next_per_atom)

        target_pred_l, target_score = _decoder_step(
            student, l_next, x_next, atom_types, num_atoms, node2graph, idx_next, sigma_next, sigma_norm_next
        )
        target_l0 = lattice_x0_estimate(l_next, target_pred_l, ac_next)
        target_x0 = coord_x0_estimate(x_next, target_score, sigma_next_per_atom)

    student_pred_l, student_score = _decoder_step(
        student, l_n, x_n, atom_types, num_atoms, node2graph, idx_n, sigma_n, sigma_norm_n
    )
    pred_l0 = lattice_x0_estimate(l_n, student_pred_l, ac_n)
    pred_x0 = coord_x0_estimate(x_n, student_score, sigma_n_per_atom)

    loss_lattice = torch.nn.functional.mse_loss(pred_l0, target_l0)
    loss_coord = torch.nn.functional.mse_loss(pred_x0, target_x0)
    loss = loss_lattice + loss_coord

    return loss, {"t_n": idx_n, "t_next": idx_next}
