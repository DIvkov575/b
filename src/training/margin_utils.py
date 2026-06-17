import torch
from torch import Tensor


def compute_margin(logits: Tensor, labels: Tensor) -> Tensor:
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels.unsqueeze(1), float("-inf"))
    other_max = masked.max(dim=1).values
    return true_logits - other_max
