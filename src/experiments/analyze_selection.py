"""Compare DPP vs centrality subgraph selections on a single graph."""

from __future__ import annotations

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def _centrality_topk(data: Data, budget_k: int) -> list[int]:
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
    k = min(budget_k, int(deg.numel()))
    if k == 0:
        return []
    topk = torch.topk(deg, k=k).indices.tolist()
    return topk


def compare_selections(data: Data, dpp_model, budget_k: int = 5) -> dict:
    """Return DPP vs centrality picks for a single graph, plus diagnostics.

    Returns a dict with:
      - dpp_selected: subgraph indices picked by DPP MAP
      - centrality_selected: node indices ranked by degree (top-k)
      - quality_scores: per-subgraph quality from the margin scorer
      - node_degrees: degree of each node in the input graph
      - overlap: count of indices appearing in both selections
    """
    dpp_model.eval()
    with torch.no_grad():
        _, info = dpp_model(data, return_margin_info=True)

    dpp_selected = list(info.get("selected_indices", []))
    quality_scores = info.get("quality_scores", torch.zeros(0))

    node_degrees = degree(data.edge_index[0], num_nodes=data.num_nodes).long()
    centrality_selected = _centrality_topk(data, budget_k)

    overlap = len(set(dpp_selected) & set(centrality_selected))

    return {
        "dpp_selected": dpp_selected,
        "centrality_selected": centrality_selected,
        "quality_scores": quality_scores.detach().cpu().tolist()
        if isinstance(quality_scores, torch.Tensor)
        else list(quality_scores),
        "node_degrees": node_degrees.detach().cpu().tolist(),
        "overlap": overlap,
    }
