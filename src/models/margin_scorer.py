import torch
import torch.nn as nn
from torch import Tensor


class MarginScorer(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, subgraph_embeddings: Tensor, graph_embedding: Tensor) -> Tensor:
        n_subgraphs = subgraph_embeddings.shape[0]
        graph_broadcast = graph_embedding.unsqueeze(0).expand(n_subgraphs, -1)
        combined = torch.cat([subgraph_embeddings, graph_broadcast], dim=-1)
        return self.score_net(combined).squeeze(-1)
