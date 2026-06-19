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

    Per the spec: R_guided(x'|x) = R_base(x'|x) * prod [p(yi|x',t)/p(yi|x,t)]^γi
    For each position l and candidate state k, we construct x' by substituting
    position l with state k, then evaluate the classifier on x'.
    """
    B, L, K = base_rates.shape
    device = x_t.device

    # log p(y|x, t) for current state (denominator) — shared across all transitions
    log_denom = torch.zeros(B, device=device)
    for clf, gamma in zip(classifiers, gammas):
        prob_current = clf.predict_prob(x_t, t)  # (B,)
        log_denom = log_denom + gamma * torch.log(prob_current.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob_current = clf.predict_prob(x_t, t)
            log_denom = log_denom - gamma * torch.log(prob_current.clamp(min=1e-8))

    # For efficiency, approximate per-transition guidance:
    # Evaluate classifier on x' for each (position, state) substitution.
    # Full version: K*L forward passes per batch. Approximate: sample a subset.
    # Here we use the "current-state" approximation for positions that don't change,
    # and per-state evaluation for the actual transition candidates.
    log_ratios = torch.zeros(B, L, K, device=device)

    for l in range(L):
        for k in range(K):
            # Construct x' by substituting position l with state k
            x_prime = x_t.clone()
            x_prime[:, l] = k
            log_num = torch.zeros(B, device=device)
            for clf, gamma in zip(classifiers, gammas):
                prob_prime = clf.predict_prob(x_prime, t)
                log_num = log_num + gamma * torch.log(prob_prime.clamp(min=1e-8))
            if avoid_classifiers:
                for clf, gamma in zip(avoid_classifiers, avoid_gammas):
                    prob_prime = clf.predict_prob(x_prime, t)
                    log_num = log_num - gamma * torch.log(prob_prime.clamp(min=1e-8))
            log_ratios[:, l, k] = log_num - log_denom

    guided = base_rates * torch.exp(log_ratios)
    return guided


def guided_rates_ctmc_fast(
    base_rates: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Fast approximation: uniform scaling (no per-transition differentiation).
    Use when K*L forward passes is too expensive.
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
    return base_rates * scale


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
