import torch

from src.models.margin_scorer import MarginScorer
from src.training.margin_utils import compute_margin


def test_compute_margin_correct_class():
    logits = torch.tensor([[3.0, 1.0, 0.5]])
    labels = torch.tensor([0])
    margin = compute_margin(logits, labels)
    assert torch.allclose(margin, torch.tensor([2.0]))


def test_compute_margin_negative():
    logits = torch.tensor([[1.0, 3.0, 0.5]])
    labels = torch.tensor([0])
    margin = compute_margin(logits, labels)
    assert torch.allclose(margin, torch.tensor([-2.0]))


def test_margin_scorer_output_shape():
    embed_dim = 16
    n_subgraphs = 10
    scorer = MarginScorer(embed_dim)
    subgraph_embeddings = torch.randn(n_subgraphs, embed_dim)
    graph_embedding = torch.randn(embed_dim)
    out = scorer(subgraph_embeddings, graph_embedding)
    assert out.shape == (n_subgraphs,)


def test_margin_scorer_gradients():
    embed_dim = 8
    n_subgraphs = 4
    scorer = MarginScorer(embed_dim)
    subgraph_embeddings = torch.randn(n_subgraphs, embed_dim, requires_grad=True)
    graph_embedding = torch.randn(embed_dim, requires_grad=True)
    out = scorer(subgraph_embeddings, graph_embedding)
    loss = out.sum()
    loss.backward()
    assert subgraph_embeddings.grad is not None
    assert graph_embedding.grad is not None
    assert subgraph_embeddings.grad.shape == subgraph_embeddings.shape
