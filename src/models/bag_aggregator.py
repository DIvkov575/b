import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BagAggregator(nn.Module):
    def __init__(self, embed_dim: int, method: str = "weighted_sum"):
        super().__init__()
        if method not in ("sum", "weighted_sum", "attention"):
            raise ValueError(f"Unknown aggregation method: {method}")
        self.embed_dim = embed_dim
        self.method = method
        if method == "attention":
            self.attn = nn.Linear(embed_dim, 1)

    def forward(self, embeddings: Tensor, weights: Tensor) -> Tensor:
        if self.method == "sum":
            return embeddings.sum(dim=0)
        if self.method == "weighted_sum":
            return (embeddings * weights.unsqueeze(-1)).sum(dim=0)
        if self.method == "attention":
            attn_logits = self.attn(embeddings).squeeze(-1)
            attn_weights = F.softmax(attn_logits, dim=0) * weights
            return (embeddings * attn_weights.unsqueeze(-1)).sum(dim=0)
        raise ValueError(f"Unknown aggregation method: {self.method}")
