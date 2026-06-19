import torch
from src.models.classifier import TimeConditionalClassifier


def guided_rates_ctmc(
    base_rates: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Apply classifier guidance to CTMC transition rates.
    Simplified: scale all rates by classifier signal at current state.
    """
    B, L, K = base_rates.shape
    log_scale = torch.zeros(B, device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        prob = clf.predict_prob(x_t, t)
        log_scale = log_scale + gamma * torch.log(prob.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob = clf.predict_prob(x_t, t)
            log_scale = log_scale - gamma * torch.log(prob.clamp(min=1e-8))

    scale = torch.exp(log_scale).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    guided = base_rates * scale
    return guided


def sample_tau_leaping(
    rates: torch.Tensor, x_t: torch.Tensor, dt: float, K: int
) -> torch.Tensor:
    """One tau-leaping step: sample transitions from rates."""
    B, L, _ = rates.shape
    trans_probs = (rates * dt).clamp(min=0)
    # Zero out self-transition
    current = x_t.unsqueeze(-1)  # (B, L, 1)
    mask = torch.zeros_like(trans_probs).scatter_(2, current, 1.0)
    trans_probs = trans_probs * (1 - mask)
    # Sample whether to transition
    total_trans = trans_probs.sum(dim=-1)  # (B, L)
    do_transition = torch.rand_like(total_trans) < total_trans
    # Where to transition
    normed = trans_probs / trans_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    flat_normed = normed.reshape(-1, K)
    new_states = torch.multinomial(flat_normed, num_samples=1).squeeze(-1).reshape(B, L)
    return torch.where(do_transition, new_states, x_t)
