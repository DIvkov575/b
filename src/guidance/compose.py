import torch
import torch.nn.functional as F
from src.models.classifier import TimeConditionalClassifier


def _classifier_log_guidance(
    clf: TimeConditionalClassifier, x_t: torch.Tensor, t: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Compute gamma * log p(y=1 | x_t, t) — scalar per batch element."""
    prob = clf.predict_prob(x_t, t)  # (B,)
    return gamma * torch.log(prob.clamp(min=1e-8))


def compose_and(
    base_logits: torch.Tensor,
    classifiers: list,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gammas: list,
    mode: str = "prob_path",
) -> torch.Tensor:
    """AND composition: multiply posteriors by product of classifier likelihoods.
    p(x_0 | x_t, A∧B) ∝ p(x_0 | x_t) * p(A|x_0)^γA * p(B|x_0)^γB
    """
    log_guidance = torch.zeros(x_t.shape[0], device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        log_guidance = log_guidance + _classifier_log_guidance(clf, x_t, t, gamma)
    return base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)


def compose_not(
    base_logits: torch.Tensor,
    clf_keep: TimeConditionalClassifier,
    clf_avoid: TimeConditionalClassifier,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gamma_a: float = 1.0,
    gamma_b: float = 1.0,
    mode: str = "prob_path",
) -> torch.Tensor:
    """NOT composition: A ∧ ¬B.
    p(x_0 | x_t, A∧¬B) ∝ p(x_0 | x_t) * p(A|x_0)^γA * p(B|x_0)^{-γB}
    """
    log_keep = _classifier_log_guidance(clf_keep, x_t, t, gamma_a)
    log_avoid = _classifier_log_guidance(clf_avoid, x_t, t, -gamma_b)
    log_guidance = log_keep + log_avoid
    return base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)


def compose_or(
    base_logits: torch.Tensor,
    classifiers: list,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gammas: list,
    mode: str = "prob_path",
) -> torch.Tensor:
    """OR composition: A ∨ B (∨ C ...).
    For 2 classifiers: inclusion-exclusion P(A∨B) = P(A) + P(B) - P(A)P(B).
    For N>2: noisy-OR approximation P(A∨B∨C) = 1 - prod(1-P(i)).
    """
    log_probs = []
    for clf, gamma in zip(classifiers, gammas):
        log_probs.append(_classifier_log_guidance(clf, x_t, t, gamma))

    if len(log_probs) == 1:
        log_or = log_probs[0]
    elif len(log_probs) == 2:
        a, b = log_probs[0], log_probs[1]
        max_ab = torch.max(a, b)
        log_or = max_ab + torch.log(
            (torch.exp(a - max_ab) + torch.exp(b - max_ab)
             - torch.exp(a + b - max_ab)).clamp(min=1e-8)
        )
    else:
        # Noisy-OR: log(1 - prod(1-p_i)) = log(1 - exp(sum(log(1-p_i))))
        # where p_i = exp(log_probs[i])
        log_complements = [torch.log1p(-torch.exp(lp).clamp(max=1.0 - 1e-7)) for lp in log_probs]
        sum_log_comp = torch.stack(log_complements).sum(dim=0)
        log_or = torch.log1p(-torch.exp(sum_log_comp).clamp(max=1.0 - 1e-7))

    return base_logits + log_or.unsqueeze(-1).unsqueeze(-1)
