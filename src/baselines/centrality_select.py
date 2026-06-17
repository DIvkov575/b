"""HyMN-style centrality-based subgraph selection baseline.

Ranks node-deletion subgraphs by the degree centrality of the deleted node and
keeps the top-k. Higher centrality of the removed node yields a more informative
subgraph (its absence perturbs the graph structure more), following the HyMN
heuristic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree

from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.base_gnn import GINEncoder


class CentralitySelect(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
        budget_k: int = 10,
    ):
        super().__init__()
        self.budget_k = budget_k
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def _centrality_scores(self, data: Data) -> list[float]:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        return deg.tolist()

    def forward(self, data: Data, return_margin_info: bool = False) -> torch.Tensor:
        data_cpu = data.cpu() if data.x.device.type != "cpu" else data
        subgraphs = node_deletion_subgraphs(data_cpu)

        if len(subgraphs) == 0:
            graph_emb = self.encoder(data)
            logits = self.classifier(graph_emb)
            if return_margin_info:
                return logits, {"selected_indices": [], "quality_scores": torch.zeros(0)}
            return logits

        scores = self._centrality_scores(data)
        scores_tensor = torch.tensor(scores, dtype=torch.float)
        k = min(self.budget_k, len(subgraphs))
        topk_idx = torch.topk(scores_tensor, k=k).indices.tolist()

        selected = [subgraphs[i] for i in topk_idx]
        device = next(self.parameters()).device
        batch = Batch.from_data_list([strip_subgraph(s) for s in selected]).to(device)
        subgraph_embeddings = self.encoder(batch)

        aggregated = subgraph_embeddings.mean(dim=0)
        logits = self.classifier(aggregated)
        if return_margin_info:
            return logits, {"selected_indices": topk_idx, "quality_scores": scores_tensor}
        return logits
