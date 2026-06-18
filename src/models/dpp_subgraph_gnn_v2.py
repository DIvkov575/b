"""V2: Fixed train/eval consistency (hard selection at both) + LOO supervision."""
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.bag_aggregator import BagAggregator
from src.models.base_gnn import GINEncoder
from src.models.dpp_selector import DPPSelector
from src.models.margin_scorer import MarginScorer


class DPPSubgraphGNNv2(nn.Module):
    """V2: Hard selection at train AND eval. LOO margin supervision."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, budget_k=10,
                 aggregation="weighted_sum"):
        super().__init__()
        self.budget_k = budget_k
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.margin_scorer = MarginScorer(hidden_dim)
        self.dpp_selector = DPPSelector(hidden_dim, budget_k)
        self.aggregator = BagAggregator(hidden_dim, aggregation)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, return_margin_info: bool = False):
        data_cpu = data.cpu() if data.x.device.type != "cpu" else data
        subgraphs = node_deletion_subgraphs(data_cpu)

        if len(subgraphs) == 0:
            graph_emb = self.encoder(data)
            logits = self.classifier(graph_emb)
            if return_margin_info:
                return logits, {"quality_scores": torch.zeros(0), "loo_scores": torch.zeros(0),
                                "selected_indices": []}
            return logits

        device = next(self.parameters()).device
        batch = Batch.from_data_list([strip_subgraph(s) for s in subgraphs]).to(device)
        subgraph_embeddings = self.encoder(batch)

        graph_context = subgraph_embeddings.mean(dim=0)
        quality_scores = self.margin_scorer(subgraph_embeddings, graph_context)

        # ALWAYS use hard selection (same at train and eval)
        selected_indices = self.dpp_selector(subgraph_embeddings.detach(), quality_scores.detach())
        sel_idx = torch.tensor(selected_indices, device=device, dtype=torch.long)
        selected_embeddings = subgraph_embeddings[sel_idx]

        # Straight-through: use selected embeddings but let gradients flow through quality_scores
        # via the DPP loss (not through selection itself)
        weights = torch.ones(len(selected_indices), device=device) / len(selected_indices)
        aggregated = self.aggregator(selected_embeddings, weights)
        logits = self.classifier(aggregated)

        if return_margin_info:
            # Compute LOO scores: margin contribution of each subgraph in selected set
            loo_scores = self._compute_loo(selected_embeddings, weights, logits)
            info = {
                "quality_scores": quality_scores,
                "loo_scores": loo_scores,
                "selected_indices": selected_indices,
                "selected_embeddings": selected_embeddings,
            }
            return logits, info
        return logits

    def _compute_loo(self, selected_embeddings, weights, full_logits):
        """Leave-one-out: how much does removing each subgraph change the prediction?"""
        k = selected_embeddings.shape[0]
        if k <= 1:
            return torch.zeros(k, device=selected_embeddings.device)

        full_logits_d = full_logits.detach()
        loo_scores = torch.zeros(k, device=selected_embeddings.device)
        for i in range(k):
            mask = torch.ones(k, dtype=torch.bool, device=selected_embeddings.device)
            mask[i] = False
            remaining = selected_embeddings[mask].detach()
            w = torch.ones(k - 1, device=selected_embeddings.device) / (k - 1)
            agg = self.aggregator(remaining, w)
            logits_without = self.classifier(agg)
            loo_scores[i] = (full_logits_d - logits_without.detach()).abs().sum()

        return loo_scores
