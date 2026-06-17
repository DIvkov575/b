import torch
from torch import Tensor


def compute_margin(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute classification margin or regression confidence.

    For classification (logits has >1 columns): margin = logit[true] - max(logit[other])
    For regression (logits has 1 column): margin = -|prediction - target| (negative error as proxy)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] == 1:
        return -torch.abs(logits.view(-1) - labels.view(-1).float())
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    other_max = masked.max(dim=1).values
    return true_logits - other_max
