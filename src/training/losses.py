import torch
import torch.nn.functional as F
from torch import Tensor


def classification_loss(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(logits, labels)


def dpp_margin_loss(quality_scores: Tensor, actual_margins: Tensor) -> Tensor:
    target = torch.sigmoid(actual_margins)
    predicted = torch.sigmoid(quality_scores)
    return F.mse_loss(predicted, target)
