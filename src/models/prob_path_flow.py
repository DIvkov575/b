import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_categorical(x0: torch.Tensor, t: float, K: int) -> torch.Tensor:
    """Linear interpolation between one-hot(x0) and uniform on the simplex.
    p_t(k) = (1-t) * one_hot(x0, k) + t * (1/K)
    """
    one_hot = F.one_hot(x0, K).float()
    uniform = torch.ones_like(one_hot) / K
    return (1.0 - t) * one_hot + t * uniform


def sample_from_categorical(x0: torch.Tensor, t: float, K: int) -> torch.Tensor:
    """Sample x_t from the interpolated categorical distribution."""
    p_t = interpolate_categorical(x0, t, K)
    flat = p_t.reshape(-1, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(x0.shape)


class ProbPathDenoiser(nn.Module):
    """Predicts p(x_1 | x_t) — the clean-data posterior given noisy observation."""
    def __init__(self, K: int, L: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.embed = nn.Embedding(K, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x_t)
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)
        h = self.net(h)
        return self.out(h)  # (B, L, K) logits for p(x_0 | x_t)


def sample_euler_step(
    x_t: torch.Tensor, posterior_probs: torch.Tensor, dt: float, K: int
) -> torch.Tensor:
    """One Euler step: mix current one-hot with predicted posterior, sample."""
    one_hot_current = F.one_hot(x_t, K).float()
    p_next = (1.0 - dt) * one_hot_current + dt * posterior_probs
    p_next = p_next.clamp(min=1e-8)
    p_next = p_next / p_next.sum(dim=-1, keepdim=True)
    B, L, _ = p_next.shape
    flat = p_next.reshape(-1, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(B, L)
