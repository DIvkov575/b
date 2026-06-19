import torch
import torch.nn.functional as F
from src.models.classifier import TimeConditionalClassifier


def guided_posterior_prob_path(
    base_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Apply classifier guidance to probability-path posterior.

    For each position l and candidate state k, construct x' with x'[l]=k
    and evaluate p(y|x', t). This gives per-position, per-state guidance.

    Falls back to fast (uniform) mode for large L*K to stay tractable.
    """
    B, L, K = base_logits.shape

    if L * K <= 512:
        return _guided_posterior_per_position(
            base_logits, x_t, t, classifiers, gammas, avoid_classifiers, avoid_gammas
        )
    else:
        return _guided_posterior_fast(
            base_logits, x_t, t, classifiers, gammas, avoid_classifiers, avoid_gammas
        )


def _guided_posterior_per_position(
    base_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Per-position guidance: evaluate classifier on each (pos, state) substitution."""
    B, L, K = base_logits.shape
    device = x_t.device
    log_guidance = torch.zeros(B, L, K, device=device)

    for l in range(L):
        for k in range(K):
            x_sub = x_t.clone()
            x_sub[:, l] = k
            log_g = torch.zeros(B, device=device)
            for clf, gamma in zip(classifiers, gammas):
                prob = clf.predict_prob(x_sub, t)
                log_g = log_g + gamma * torch.log(prob.clamp(min=1e-8))
            if avoid_classifiers:
                for clf, gamma in zip(avoid_classifiers, avoid_gammas):
                    prob = clf.predict_prob(x_sub, t)
                    log_g = log_g - gamma * torch.log(prob.clamp(min=1e-8))
            log_guidance[:, l, k] = log_g

    return base_logits + log_guidance


def _guided_posterior_fast(
    base_logits: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    classifiers: list,
    gammas: list,
    avoid_classifiers: list = None,
    avoid_gammas: list = None,
) -> torch.Tensor:
    """Fast approximation: single scalar guidance broadcast to all positions."""
    log_guidance = torch.zeros(x_t.shape[0], device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        prob = clf.predict_prob(x_t, t)
        log_guidance = log_guidance + gamma * torch.log(prob.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob = clf.predict_prob(x_t, t)
            log_guidance = log_guidance - gamma * torch.log(prob.clamp(min=1e-8))

    return base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)
