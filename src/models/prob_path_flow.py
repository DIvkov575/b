import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate_categorical(x0: torch.Tensor, t, K: int) -> torch.Tensor:
    """Linear interpolation between one-hot(x0) and uniform on the simplex.
    p_t(k) = (1-t) * one_hot(x0, k) + t * (1/K)
    t can be float or (B,) tensor. x0: (B, L) or (L,).
    """
    one_hot = F.one_hot(x0, K).float()
    uniform = torch.ones_like(one_hot) / K
    if isinstance(t, torch.Tensor) and t.dim() >= 1:
        # t: (B,) -> (B, 1, 1) for broadcasting over (B, L, K)
        t_expanded = t.float().reshape(-1, 1, 1) if x0.dim() == 2 else t.float()
    else:
        t_expanded = t
    return (1.0 - t_expanded) * one_hot + t_expanded * uniform


def sample_from_categorical(x0: torch.Tensor, t, K: int) -> torch.Tensor:
    """Sample x_t from the interpolated categorical distribution.
    t can be float or (B,) tensor. x0: (B, L) or (L,).
    """
    p_t = interpolate_categorical(x0, t, K)
    flat = p_t.reshape(-1, K)
    samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return samples.reshape(x0.shape)


class ProbPathDenoiser(nn.Module):
    """Predicts p(x_1 | x_t) — the clean-data posterior given noisy observation.
    Uses Transformer encoder for inter-position communication.
    """
    def __init__(self, K: int, L: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.embed = nn.Embedding(K, hidden_dim)
        self.pos_embed = nn.Embedding(L, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)
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
