import torch
from torch_geometric.data import Data, Batch

from src.models.base_gnn import GINEncoder


def _make_graph(num_nodes=6, in_dim=8, num_edges=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.randint(0, num_nodes, (num_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_edges,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index)


def test_gin_output_shape():
    in_dim, hidden_dim, out_dim = 8, 16, 32
    model = GINEncoder(in_dim, hidden_dim, out_dim, num_layers=4).eval()
    data = _make_graph(in_dim=in_dim)

    with torch.no_grad():
        out = model(data)

    assert out.shape == (out_dim,), f"expected ({out_dim},), got {tuple(out.shape)}"


def test_gin_batch():
    in_dim, hidden_dim, out_dim = 8, 16, 32
    model = GINEncoder(in_dim, hidden_dim, out_dim, num_layers=4).eval()
    graphs = [_make_graph(num_nodes=5 + i, in_dim=in_dim, seed=i) for i in range(4)]
    batch = Batch.from_data_list(graphs)

    with torch.no_grad():
        out = model(batch)

    assert out.shape == (4, out_dim), f"expected (4, {out_dim}), got {tuple(out.shape)}"


def test_gin_deterministic():
    in_dim, hidden_dim, out_dim = 8, 16, 32
    model = GINEncoder(in_dim, hidden_dim, out_dim, num_layers=4).eval()
    data = _make_graph(in_dim=in_dim, seed=42)

    with torch.no_grad():
        out1 = model(data)
        out2 = model(data)

    assert torch.allclose(out1, out2), "GINEncoder must be deterministic in eval mode"
