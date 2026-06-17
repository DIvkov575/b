import torch

from src.models.dpp_selector import DPPSelector


def test_dpp_selector_output_size():
    torch.manual_seed(0)
    selector = DPPSelector(embed_dim=8, budget_k=3)
    embeddings = torch.randn(10, 8)
    quality = torch.randn(10)
    selected = selector(embeddings, quality)
    assert len(selected) == 3


def test_dpp_selector_indices_valid():
    torch.manual_seed(1)
    selector = DPPSelector(embed_dim=8, budget_k=4)
    embeddings = torch.randn(12, 8)
    quality = torch.randn(12)
    selected = selector(embeddings, quality)
    assert all(0 <= i < 12 for i in selected)
    assert len(set(selected)) == len(selected)


def test_dpp_selector_budget_exceeds_candidates():
    torch.manual_seed(2)
    selector = DPPSelector(embed_dim=8, budget_k=10)
    embeddings = torch.randn(5, 8)
    quality = torch.randn(5)
    selected = selector(embeddings, quality)
    assert len(selected) == 5
    assert sorted(selected) == [0, 1, 2, 3, 4]


def test_dpp_selector_differentiable_relaxation():
    torch.manual_seed(3)
    selector = DPPSelector(embed_dim=8, budget_k=3, temperature=0.5)
    embeddings = torch.randn(10, 8, requires_grad=True)
    quality = torch.randn(10, requires_grad=True)
    soft = selector.soft_select(embeddings, quality)
    assert soft.shape == (10,)
    loss = soft.sum()
    loss.backward()
    assert embeddings.grad is not None
    assert quality.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.isfinite(quality.grad).all()
