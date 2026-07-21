"""L35 consistency-distillation training step for DiffCSP's dual-track (lattice +
fractional-coordinate) denoiser.

Composes src/l35/consistency_distill.py's boundary-condition math with a real
decoder's calling convention. Two teacher-output conventions are load-bearing
and were taken directly from diffcsp/pl_modules/diffusion.py, not re-derived:
  - pred_l IS the epsilon prediction directly (forward(): loss_lattice against rand_l).
  - pred_x is a NORMALIZED score; sample() rescales it via `pred_x * sqrt(sigma_norm)`
    before using it as the true score fed to the annealed-Langevin update.

Consistency distillation against an EMA target network (Song et al. 2023,
"Consistency Models", Eq. 8), not the live student itself. The original
paper explicitly measured that setting the target network theta_minus equal
to the live online network theta (rather than an EMA of its history)
destabilizes training; theta_minus == theta is expected only asymptotically,
at convergence. A first version of this module used the live student as its
own target (matching L37's training_step.py structure) and reproduced
exactly this instability empirically: a 256-structure/30-epoch local pilot
stayed bounded, but a 27,136-structure/15-epoch full run diverged (loss
grew ~300x even with gradient clipping) -- consistent with a slow systematic
bias compounding over the ~13x-larger step count, which per-step gradient
clipping (bounding update magnitude, not directional bias) cannot fix.
See docs/L35_PIPELINE_SPEC.md for the full incident writeup.

Follows Song et al. 2023 Algorithm 2 (Consistency Distillation) exactly:
starting from the REAL, clean batch, forward-noise it up to the noisier of
the sampled index pair (idx_next), run the teacher there and take one ODE
step down to the less-noisy index (idx_n) to build the target input, then
compare the student's consistency function at (noised-to-idx_next, idx_next)
against the target network's at (teacher-denoised-to-idx_n, idx_n). A first
version of this module skipped forward-noising entirely -- it fed the RAW,
CLEAN batch straight into every decoder call (student at idx_n, target at
idx_next -- also the wrong way around), mislabeled with fake timestep
indices. That version ran a real 27,136-structure/15-epoch EC2 training job
to completion (loss climbed smoothly and continuously, ~1,248 to ~200,000,
while weight_norm stayed frozen, since lr=1e-6 + grad clipping bounded each
step tightly) and the resulting "distilled" student scored WORSE
(match_rate=0.0, valid=0.48) than simply truncating the UNTRAINED teacher to
the same 8-step count (match_rate=0.18, valid=0.90) -- consistent with
training against a target with no real relationship to the input, since the
decoder was always being asked to denoise data that was already clean.
"""
import torch

from src.l35.consistency_distill import (
    coord_forward_noise,
    coord_pfode_step,
    coord_x0_estimate,
    lattice_ddim_step,
    lattice_forward_noise,
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
    teacher, student, target_network, batch, num_steps, beta_scheduler, sigma_scheduler, generator=None
):
    """One consistency-distillation training step over a toy/real DiffCSP batch.

    Args:
        teacher: frozen decoder, callable as CSPNet (time_emb, atom_types, frac_coords,
            lattices, num_atoms, node2graph) -> (pred_l, pred_x). No gradient flows into
            it regardless of its own requires_grad state (wrapped in no_grad).
        student: trainable decoder with the same calling convention.
        target_network: the "target"/online-EMA network (Song et al. 2023 Eq. 8) used
            to compute the training target -- must NOT be the live student itself (see
            module docstring for why). No gradient flows into it regardless of its own
            requires_grad state (wrapped in no_grad, same guarantee as teacher). Callers
            own updating it (e.g. an EMA step after each optimizer.step()).
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

    l_0 = batch["lattices"]
    x_0 = batch["frac_coords"]

    # Fresh Gaussian noise, drawn on CPU via the (optional) generator then
    # moved to the batch's device -- same generator-stays-CPU convention
    # sample_index_pair already uses, avoiding the CPU/CUDA generator-device
    # mismatch caught once already in evaluate.py's sampler.
    rand_l = torch.randn(l_0.shape, generator=generator).to(l_0.device)
    rand_x = torch.randn(x_0.shape, generator=generator).to(x_0.device)

    # Forward-noise the real, clean batch up to the NOISIER of the sampled
    # index pair (idx_next), matching diffusion.py forward()'s exact noising
    # formula (Song et al. 2023 Algorithm 2: x_{t_{n+1}} ~ N(x, t_{n+1}^2 I)).
    # An earlier version skipped this and fed the clean batch itself into
    # every decoder call under a fake timestep label -- see module docstring.
    l_next = lattice_forward_noise(l_0, rand_l, ac_next)
    x_next = coord_forward_noise(x_0, rand_x, sigma_next_per_atom)

    with torch.no_grad():
        # Teacher takes ONE deterministic ODE step from idx_next down to the
        # less-noisy idx_n (the numerical solver step in Algorithm 2),
        # building \hat{x}_{t_n}^\phi -- the target network's input.
        pred_l_next, score_next = _decoder_step(
            teacher, l_next, x_next, atom_types, num_atoms, node2graph, idx_next, sigma_next, sigma_norm_next
        )
        l_n_hat = lattice_ddim_step(l_next, pred_l_next, ac_next, ac_n)
        x_n_hat = coord_pfode_step(x_next, score_next, sigma_next_per_atom, sigma_n_per_atom)

        target_pred_l, target_score = _decoder_step(
            target_network, l_n_hat, x_n_hat, atom_types, num_atoms, node2graph, idx_n, sigma_n, sigma_norm_n
        )
        target_l0 = lattice_x0_estimate(l_n_hat, target_pred_l, ac_n)
        target_x0 = coord_x0_estimate(x_n_hat, target_score, sigma_n_per_atom)

    # Student's consistency function is evaluated at the SAME noised point
    # the teacher started its solver step from (x_{t_{n+1}}, t_{n+1}) --
    # not at idx_n, and not on clean data (see module docstring).
    student_pred_l, student_score = _decoder_step(
        student, l_next, x_next, atom_types, num_atoms, node2graph, idx_next, sigma_next, sigma_norm_next
    )
    pred_l0 = lattice_x0_estimate(l_next, student_pred_l, ac_next)
    pred_x0 = coord_x0_estimate(x_next, student_score, sigma_next_per_atom)

    loss_lattice = torch.nn.functional.mse_loss(pred_l0, target_l0)
    loss_coord = torch.nn.functional.mse_loss(pred_x0, target_x0)
    loss = loss_lattice + loss_coord

    return loss, {"t_n": idx_n, "t_next": idx_next}
