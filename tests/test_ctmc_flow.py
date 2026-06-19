import torch
from src.models.ctmc_flow import (
    uniform_rate_matrix,
    noise_sequence,
    CTMCDenoiser,
    compute_rate_from_denoiser,
)


def test_uniform_rate_matrix():
    R = uniform_rate_matrix(K=8)
    assert R.shape == (8, 8)
    assert torch.allclose(R.sum(dim=1), torch.zeros(8), atol=1e-6)
    assert (R[torch.eye(8, dtype=torch.bool) == False] >= 0).all()


def test_noise_sequence():
    x0 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    x_t = noise_sequence(x0, t=0.0, K=8)
    assert (x_t == x0).all()

    x_t_noisy = noise_sequence(x0, t=1.0, K=8)
    assert x_t_noisy.shape == x0.shape
    assert x_t_noisy.min() >= 0 and x_t_noisy.max() <= 7


def test_ctmc_denoiser_forward():
    model = CTMCDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 32, 8)


def test_compute_rate_from_denoiser():
    model = CTMCDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (1, 32))
    t = torch.tensor([0.5])
    rates = compute_rate_from_denoiser(model, x_t, t, K=8)
    assert rates.shape == (1, 32, 8)
    # rate to current state should be <= 0 (negative diagonal)
    for l in range(32):
        assert rates[0, l, x_t[0, l].item()].item() <= 0
