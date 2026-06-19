import torch
from src.models.prob_path_flow import (
    interpolate_categorical,
    ProbPathDenoiser,
    sample_euler_step,
)


def test_interpolate_categorical():
    x0 = torch.tensor([0, 1, 2, 3])
    K = 8
    # t=0 → one-hot on x0
    p_t = interpolate_categorical(x0, t=0.0, K=K)
    assert p_t.shape == (4, K)
    assert torch.allclose(p_t.sum(dim=-1), torch.ones(4))
    for i in range(4):
        assert p_t[i, x0[i]].item() > 0.99

    # t=1 → uniform
    p_t = interpolate_categorical(x0, t=1.0, K=K)
    expected = torch.ones(4, K) / K
    assert torch.allclose(p_t, expected, atol=0.01)


def test_prob_path_denoiser_forward():
    model = ProbPathDenoiser(K=8, L=32, hidden_dim=64, num_layers=2)
    x_t = torch.randint(0, 8, (4, 32))
    t = torch.tensor([0.5, 0.3, 0.8, 0.1])
    logits = model(x_t, t)
    assert logits.shape == (4, 32, 8)


def test_sample_euler_step():
    K = 8
    posterior = torch.zeros(2, 4, K)
    posterior[:, :, 0] = 10.0
    posterior = torch.softmax(posterior, dim=-1)
    x_t = torch.randint(0, K, (2, 4))
    x_next = sample_euler_step(x_t, posterior, dt=0.1, K=K)
    assert x_next.shape == (2, 4)
    assert x_next.min() >= 0 and x_next.max() < K
