import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data

from src.data.subgraph_policies import (
    edge_deletion_subgraphs,
    ego_subgraphs,
    node_deletion_subgraphs,
)
from src.models.bag_aggregator import BagAggregator
from src.models.base_gnn import GINEncoder
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer
from src.training.margin_utils import compute_margin


POLICIES = {
    "node_deletion": node_deletion_subgraphs,
    "edge_deletion": edge_deletion_subgraphs,
    "ego": ego_subgraphs,
}


class DPPSubgraphGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
        budget_k: int = 10,
        policy: str = "node_deletion",
        aggregation: str = "weighted_sum",
    ):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}. Available: {list(POLICIES)}")
        self.policy_name = policy
        self.policy_fn = POLICIES[policy]
        self.budget_k = budget_k

        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.margin_scorer = MarginScorer(hidden_dim)
        self.dpp_selector = DPPSelector(hidden_dim, budget_k)
        self.aggregator = BagAggregator(hidden_dim, aggregation)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, return_margin_info: bool = False):
        subgraphs = self.policy_fn(data)

        if len(subgraphs) == 0:
            graph_emb = self.encoder(data)
            logits = self.classifier(graph_emb)
            if return_margin_info:
                info = {
                    "selected_indices": [],
                    "quality_scores": torch.zeros(0, device=logits.device),
                    "subgraph_embeddings": torch.zeros(0, graph_emb.shape[-1], device=logits.device),
                }
                if getattr(data, "y", None) is not None:
                    info["margin"] = compute_margin(logits.unsqueeze(0), data.y.view(-1))
                return logits, info
            return logits

        batch = Batch.from_data_list(subgraphs)
        subgraph_embeddings = self.encoder(batch)

        graph_context = subgraph_embeddings.mean(dim=0)
        quality_scores = self.margin_scorer(subgraph_embeddings, graph_context)

        if self.training:
            soft_weights = self.dpp_selector.soft_select(subgraph_embeddings, quality_scores)
            aggregated = self.aggregator(subgraph_embeddings, soft_weights)
            selected_indices = self.dpp_selector(subgraph_embeddings.detach(), quality_scores.detach())
        else:
            selected_indices = self.dpp_selector(subgraph_embeddings, quality_scores)
            sel_idx = torch.tensor(selected_indices, device=subgraph_embeddings.device, dtype=torch.long)
            selected_embeddings = subgraph_embeddings.index_select(0, sel_idx)
            uniform_weights = torch.ones(len(selected_indices), device=subgraph_embeddings.device)
            aggregated = self.aggregator(selected_embeddings, uniform_weights)

        logits = self.classifier(aggregated)

        if return_margin_info:
            info = {
                "selected_indices": selected_indices,
                "quality_scores": quality_scores,
                "subgraph_embeddings": subgraph_embeddings,
            }
            if getattr(data, "y", None) is not None:
                info["margin"] = compute_margin(logits.unsqueeze(0), data.y.view(-1))
            return logits, info
        return logits
