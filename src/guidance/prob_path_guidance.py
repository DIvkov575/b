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
    p_guided(x_0|x_t) ∝ p_base(x_0|x_t) * prod p(yi|x_t,t)^γi / prod p(yj|x_t,t)^γj
    """
    log_guidance = torch.zeros(x_t.shape[0], device=x_t.device)
    for clf, gamma in zip(classifiers, gammas):
        prob = clf.predict_prob(x_t, t)
        log_guidance = log_guidance + gamma * torch.log(prob.clamp(min=1e-8))

    if avoid_classifiers:
        for clf, gamma in zip(avoid_classifiers, avoid_gammas):
            prob = clf.predict_prob(x_t, t)
            log_guidance = log_guidance - gamma * torch.log(prob.clamp(min=1e-8))

    guided = base_logits + log_guidance.unsqueeze(-1).unsqueeze(-1)
    return guided
