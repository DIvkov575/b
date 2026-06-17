import torch

from src.models.bag_aggregator import BagAggregator


def test_bag_aggregator_sum():
    agg = BagAggregator(embed_dim=16, method="sum")
    embeddings = torch.randn(5, 16)
    weights = torch.ones(5)
    out = agg(embeddings, weights)
    assert out.shape == (16,)
    assert torch.allclose(out, embeddings.sum(dim=0))


def test_bag_aggregator_weighted():
    agg = BagAggregator(embed_dim=16, method="weighted_sum")
    embeddings = torch.randn(5, 16)
    weights = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    out = agg(embeddings, weights)
    assert out.shape == (16,)
    assert torch.allclose(out, embeddings[0])


def test_bag_aggregator_attention():
    agg = BagAggregator(embed_dim=16, method="attention")
    embeddings = torch.randn(5, 16)
    weights = torch.rand(5)
    out = agg(embeddings, weights)
    assert out.shape == (16,)
