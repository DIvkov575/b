import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.data.subgraph_policies import node_deletion_subgraphs, strip_subgraph
from src.models.base_gnn import GINEncoder


class ESANFull(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 4,
    ):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden_dim, hidden_dim, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data, return_info: bool = False, return_margin_info: bool = False):
        data_cpu = data.cpu() if data.x.device.type != "cpu" else data
        subgraphs = node_deletion_subgraphs(data_cpu)

        if len(subgraphs) == 0:
            graph_emb = self.encoder(data)
            logits = self.classifier(graph_emb)
            if return_margin_info or return_info:
                return logits, {"n_subgraphs": 0, "selected_indices": [], "quality_scores": torch.zeros(0)}
            return logits

        device = data.x.device
        batch = Batch.from_data_list([strip_subgraph(s) for s in subgraphs]).to(device)
        subgraph_embeddings = self.encoder(batch)
        bag_emb = subgraph_embeddings.mean(dim=0)
        logits = self.classifier(bag_emb)

        if return_margin_info or return_info:
            return logits, {"n_subgraphs": len(subgraphs), "selected_indices": [], "quality_scores": torch.zeros(0)}
        return logits
