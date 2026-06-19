import torch
from src.models.conditional_flow import ConditionalProbPathFlow
from src.guidance.cfg_compose import (
    cfg_single,
    cfg_and,
    cfg_not,
    cfg_or,
    cfg_sample,
)


def _make_model():
    return ConditionalProbPathFlow(K=4, L=8, n_conditions=3, hidden_dim=32, num_layers=1)


def test_cfg_single():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_single(model, x_t, t, condition_idx=0, n_conditions=3, w=1.0)
    assert logits.shape == (2, 8, 4)


def test_cfg_and():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_and(model, x_t, t, condition_idxs=[0, 1], n_conditions=3, ws=[1.0, 1.0])
    assert logits.shape == (2, 8, 4)


def test_cfg_not():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_not(model, x_t, t, keep_idx=0, avoid_idx=2, n_conditions=3, w_keep=1.0, w_avoid=1.0)
    assert logits.shape == (2, 8, 4)


def test_cfg_or():
    model = _make_model()
    model.eval()
    x_t = torch.randint(0, 4, (2, 8))
    t = torch.tensor([0.5, 0.5])
    logits = cfg_or(model, x_t, t, condition_idxs=[0, 1], n_conditions=3, ws=[1.0, 1.0])
    assert logits.shape == (2, 8, 4)


def test_cfg_sample():
    model = _make_model()
    model.eval()
    samples = cfg_sample(model, n=10, K=4, L=8, n_conditions=3,
                         condition_idxs=[0], ws=[2.0], num_steps=10)
    assert samples.shape == (10, 8)
    assert samples.min() >= 0 and samples.max() < 4
