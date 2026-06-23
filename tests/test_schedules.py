import torch
from src.training.schedules import (
    uniform_schedule,
    bell_schedule,
    optimal_schedule,
    sample_timesteps,
)


def test_uniform_schedule():
    weights = uniform_schedule(n_bins=100)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_bell_schedule():
    weights = bell_schedule(n_bins=100, peak=0.5, width=0.2)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert weights[45:55].sum() > weights[0:10].sum()


def test_optimal_schedule():
    it_values = torch.zeros(100)
    it_values[40:60] = 1.0
    weights = optimal_schedule(it_values)
    assert weights.shape == (100,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-5)
    assert weights[40:60].sum() > 0.9


def test_sample_timesteps():
    weights = bell_schedule(n_bins=100)
    t = sample_timesteps(batch_size=64, weights=weights)
    assert t.shape == (64,)
    assert (t >= 0).all() and (t <= 1).all()
