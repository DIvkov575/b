import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Batch, Data


class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer in range(num_layers):
            input_dim = in_dim if layer == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        single = not hasattr(data, "batch") or data.batch is None
        if single:
            data = Batch.from_data_list([data])

        x, edge_index, batch = data.x.float(), data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        graph_emb = global_add_pool(x, batch)
        out = self.proj(graph_emb)

        if single:
            out = out.squeeze(0)
        return out
