import torch
from torch_geometric.data import Data

from src.models.dpp_subgraph_gnn import DPPSubgraphGNN


def _make_cycle_graph(num_nodes: int = 4, in_dim: int = 7, seed: int = 0) -> Data:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_nodes, in_dim, generator=g)
    src = torch.tensor([i for i in range(num_nodes)] + [(i + 1) % num_nodes for i in range(num_nodes)], dtype=torch.long)
    dst = torch.tensor([(i + 1) % num_nodes for i in range(num_nodes)] + [i for i in range(num_nodes)], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


def test_forward_shape():
    torch.manual_seed(0)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = DPPSubgraphGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        budget_k=2,
        policy="node_deletion",
        aggregation="weighted_sum",
    ).eval()
    data = _make_cycle_graph(in_dim=in_dim)

    with torch.no_grad():
        out = model(data)

    assert out.shape == (out_dim,), f"expected ({out_dim},), got {tuple(out.shape)}"


def test_forward_with_labels_returns_margin():
    torch.manual_seed(1)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = DPPSubgraphGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        budget_k=2,
        policy="node_deletion",
        aggregation="weighted_sum",
    ).eval()
    data = _make_cycle_graph(in_dim=in_dim)
    data.y = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        out, info = model(data, return_margin_info=True)

    assert out.shape == (out_dim,)
    assert "selected_indices" in info
    assert "quality_scores" in info
    assert "subgraph_embeddings" in info
    assert "margin" in info
    assert len(info["selected_indices"]) == 2
    assert info["quality_scores"].shape == (4,)
    assert info["subgraph_embeddings"].shape == (4, hidden_dim)
    assert torch.isfinite(info["margin"]).all()


def test_backward():
    torch.manual_seed(2)
    in_dim, hidden_dim, out_dim = 7, 16, 3
    model = DPPSubgraphGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_layers=2,
        budget_k=2,
        policy="node_deletion",
        aggregation="weighted_sum",
    )
    model.train()
    data = _make_cycle_graph(in_dim=in_dim)
    data.y = torch.tensor([0], dtype=torch.long)

    logits = model(data)
    loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), data.y)
    loss.backward()

    grad_params = []
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0:
                grad_params.append(name)
            else:
                no_grad_params.append(name)

    assert len(grad_params) > 0, "expected at least one parameter to receive gradient"
    assert len(no_grad_params) == 0, f"params without gradient: {no_grad_params}"
