import torch
from torch_geometric.data import Data

from src.baselines.esan_full import ESANFull
from src.baselines.esan_uniform import ESANUniform


def _make_cycle_graph(num_nodes: int = 4, in_dim: int = 7, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.tensor(
        [i for i in range(num_nodes)] + [(i + 1) % num_nodes for i in range(num_nodes)],
        dtype=torch.long,
    )
    dst = torch.tensor(
        [(i + 1) % num_nodes for i in range(num_nodes)] + [i for i in range(num_nodes)],
        dtype=torch.long,
    )
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_esan_uniform_output():
    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = ESANUniform(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        budget_k=2,
    ).eval()
    data = _make_cycle_graph(in_dim=in_dim)

    with torch.no_grad():
        out = model(data)

    assert out.shape == (out_dim,), f"expected ({out_dim},), got {tuple(out.shape)}"


def test_esan_full_output():
    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = ESANFull(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
    ).eval()
    data = _make_cycle_graph(in_dim=in_dim)

    with torch.no_grad():
        out = model(data)

    assert out.shape == (out_dim,), f"expected ({out_dim},), got {tuple(out.shape)}"


def test_esan_full_uses_all_subgraphs():
    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = ESANFull(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
    ).eval()
    data = _make_cycle_graph(num_nodes=4, in_dim=in_dim)

    with torch.no_grad():
        _, info = model(data, return_info=True)

    assert info["n_subgraphs"] == 4
