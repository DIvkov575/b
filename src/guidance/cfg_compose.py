import torch
import torch.nn.functional as F
from src.models.conditional_flow import ConditionalProbPathFlow
from src.models.prob_path_flow import sample_euler_step


def _get_logits(model: ConditionalProbPathFlow, x_t: torch.Tensor,
                t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x_t, t, cond, cfg_dropout_prob=0.0)


def _null_cond(B: int, n_conditions: int, device) -> torch.Tensor:
    return torch.zeros(B, n_conditions, device=device)


def _one_hot_cond(B: int, condition_idx: int, n_conditions: int, device) -> torch.Tensor:
    cond = torch.zeros(B, n_conditions, device=device)
    cond[:, condition_idx] = 1.0
    return cond


def cfg_single(model: ConditionalProbPathFlow, x_t: torch.Tensor,
               t: torch.Tensor, condition_idx: int, n_conditions: int,
               w: float = 1.0) -> torch.Tensor:
    """Single-condition CFG: logits_cfg = (1+w)*logits_c - w*logits_uncond"""
    B = x_t.shape[0]
    device = x_t.device
    cond = _one_hot_cond(B, condition_idx, n_conditions, device)
    null = _null_cond(B, n_conditions, device)
    logits_cond = _get_logits(model, x_t, t, cond)
    logits_uncond = _get_logits(model, x_t, t, null)
    return (1 + w) * logits_cond - w * logits_uncond


def cfg_and(model: ConditionalProbPathFlow, x_t: torch.Tensor,
            t: torch.Tensor, condition_idxs: list, n_conditions: int,
            ws: list) -> torch.Tensor:
    """AND composition: product of experts in log-probability space.
    logits = logits_uncond + sum_i w_i * (logits_ci - logits_uncond)
    """
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    composed = logits_uncond.clone()
    for idx, w in zip(condition_idxs, ws):
        cond = _one_hot_cond(B, idx, n_conditions, device)
        logits_c = _get_logits(model, x_t, t, cond)
        composed = composed + w * (logits_c - logits_uncond)

    return composed


def cfg_not(model: ConditionalProbPathFlow, x_t: torch.Tensor,
            t: torch.Tensor, keep_idx: int, avoid_idx: int,
            n_conditions: int, w_keep: float = 1.0,
            w_avoid: float = 1.0) -> torch.Tensor:
    """NOT composition: A and not B. Boost A, suppress B.
    logits = logits_uncond + w_keep*(logits_A - logits_uncond) - w_avoid*(logits_B - logits_uncond)
    """
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    cond_keep = _one_hot_cond(B, keep_idx, n_conditions, device)
    cond_avoid = _one_hot_cond(B, avoid_idx, n_conditions, device)
    logits_keep = _get_logits(model, x_t, t, cond_keep)
    logits_avoid = _get_logits(model, x_t, t, cond_avoid)

    return (logits_uncond
            + w_keep * (logits_keep - logits_uncond)
            - w_avoid * (logits_avoid - logits_uncond))


def cfg_or(model: ConditionalProbPathFlow, x_t: torch.Tensor,
           t: torch.Tensor, condition_idxs: list, n_conditions: int,
           ws: list) -> torch.Tensor:
    """OR composition: element-wise max of guided logits."""
    B = x_t.shape[0]
    device = x_t.device
    null = _null_cond(B, n_conditions, device)
    logits_uncond = _get_logits(model, x_t, t, null)

    guided_logits = []
    for idx, w in zip(condition_idxs, ws):
        cond = _one_hot_cond(B, idx, n_conditions, device)
        logits_c = _get_logits(model, x_t, t, cond)
        cfg_logits = logits_uncond + w * (logits_c - logits_uncond)
        guided_logits.append(cfg_logits)

    stacked = torch.stack(guided_logits, dim=0)
    return stacked.max(dim=0).values


@torch.no_grad()
def cfg_sample(model: ConditionalProbPathFlow, n: int, K: int, L: int,
               n_conditions: int, condition_idxs: list, ws: list,
               num_steps: int = 50, mode: str = "and") -> torch.Tensor:
    """Generate samples with CFG composition."""
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)

        if mode == "and":
            logits = cfg_and(model, x_t, t, condition_idxs, n_conditions, ws)
        elif mode == "or":
            logits = cfg_or(model, x_t, t, condition_idxs, n_conditions, ws)
        elif mode == "single":
            logits = cfg_single(model, x_t, t, condition_idxs[0], n_conditions, ws[0])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        posterior = F.softmax(logits, dim=-1)
        x_t = sample_euler_step(x_t, posterior, dt, K)

    return x_t


@torch.no_grad()
def cfg_sample_not(model: ConditionalProbPathFlow, n: int, K: int, L: int,
                   n_conditions: int, keep_idx: int, avoid_idx: int,
                   w_keep: float, w_avoid: float,
                   num_steps: int = 50) -> torch.Tensor:
    """Generate samples with NOT composition."""
    device = next(model.parameters()).device
    x_t = torch.randint(0, K, (n, L), device=device)
    dt = 1.0 / num_steps

    model.eval()
    for step in range(num_steps):
        t_val = 1.0 - (step + 1) * dt
        t = torch.full((n,), t_val, device=device)
        logits = cfg_not(model, x_t, t, keep_idx, avoid_idx, n_conditions, w_keep, w_avoid)
        posterior = F.softmax(logits, dim=-1)
        x_t = sample_euler_step(x_t, posterior, dt, K)

    return x_t
