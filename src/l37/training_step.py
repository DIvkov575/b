"""L37 consistency-distillation training step.

Distills a frozen MDGen teacher's flow-matching ODE to a few-step student by consistency
distillation (Song et al. 2023 lineage): the teacher's velocity field takes one Euler step
from x_n (at t_n) to an estimate of x_next (at t_next); the student is trained so its own
consistency-function boundary-condition output at t_n matches a stop-gradient copy of that
same boundary-condition output evaluated at t_next. Both sides use the SAME student network
(self-distillation), matching consistency distillation rather than progressive distillation.
"""
import torch

from src.l37.consistency_distill import consistency_output, euler_step, masked_mse, sample_timestep_pair


def consistency_distillation_loss(teacher, student, x_n, num_steps, mask, generator=None):
    """One consistency-distillation training step.

    Args:
        teacher: frozen velocity model, callable as teacher(x, t) -> velocity. No gradient
            flows into it regardless of its own requires_grad state (wrapped in no_grad).
        student: trainable raw-output model, callable as student(x, t) -> raw_output.
        x_n: input sample at time t_n, shape (batch, ...).
        num_steps: number of steps on the {0, 1/N, ..., 1} timestep grid to sample from.
        mask: same shape as x_n; 1 for valid positions, 0 for masked-out positions.
        generator: optional torch.Generator for reproducible timestep sampling.

    Returns:
        (loss, aux) where aux is a dict with the sampled 't_n' and 't_next' tensors.
    """
    batch_size = x_n.shape[0]
    t_n, t_next = sample_timestep_pair(num_steps, batch_size, generator=generator)
    t_n = t_n.to(x_n.device)
    t_next = t_next.to(x_n.device)

    with torch.no_grad():
        velocity = teacher(x_n, t_n)
        x_next = euler_step(x_n, t_n, t_next, velocity)
        target_raw = student(x_next, t_next)
        target = consistency_output(target_raw, x_next, t_next)

    student_raw = student(x_n, t_n)
    pred = consistency_output(student_raw, x_n, t_n)

    loss = masked_mse(pred, target, mask)
    return loss, {"t_n": t_n, "t_next": t_next}
