import torch
from src.models.classifier import TimeConditionalClassifier
from src.guidance.compose import compose_and, compose_not, compose_or
from src.guidance.ctmc_guidance import guided_rates_ctmc
from src.guidance.prob_path_guidance import guided_posterior_prob_path


def _make_classifier(K=8, L=32):
    return TimeConditionalClassifier(K=K, L=L, hidden_dim=32, num_layers=1)


def test_compose_and_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_and(
        base_logits, [clf_a, clf_b], x_t, t, gammas=[1.0, 1.0], mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)
    probs = torch.softmax(guided, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4, 32), atol=1e-5)


def test_compose_not_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_not(
        base_logits, clf_a, clf_b, x_t, t, gamma_a=1.0, gamma_b=1.0, mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)


def test_compose_or_prob_path():
    clf_a = _make_classifier()
    clf_b = _make_classifier()
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5] * 4)
    base_logits = torch.randn(4, 32, 8)
    guided = compose_or(
        base_logits, [clf_a, clf_b], x_t, t, gammas=[1.0, 1.0], mode="prob_path"
    )
    assert guided.shape == (4, 32, 8)


def test_guided_rates_ctmc():
    clf = _make_classifier()
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    base_rates = torch.rand(2, 32, 8)
    guided = guided_rates_ctmc(base_rates, x_t, t, [clf], gammas=[1.0])
    assert guided.shape == (2, 32, 8)


def test_guided_posterior_prob_path():
    clf = _make_classifier()
    x_t = torch.randint(0, 8, (2, 32))
    t = torch.tensor([0.5, 0.5])
    base_logits = torch.randn(2, 32, 8)
    guided = guided_posterior_prob_path(base_logits, x_t, t, [clf], gammas=[1.0])
    assert guided.shape == (2, 32, 8)
    probs = torch.softmax(guided, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 32), atol=1e-5)
