import torch
import torch.nn as nn
import torch.nn.functional as F


def uniform_rate_matrix(K: int) -> torch.Tensor:
    """Create a K×K uniform rate matrix where off-diagonal entries are 1/(K-1) and rows sum to 0."""
    R = torch.ones(K, K) / (K - 1)
    R.fill_diagonal_(0.0)
    R -= torch.diag(R.sum(dim=1))
    return R


def noise_sequence(x0: torch.Tensor, t, K: int) -> torch.Tensor:
    """Noise clean sequences x0 at time t. Supports batched operation.
    t can be a float (applied to all) or a (B,) tensor (per-sample time).
    x0: (B, L) or (L,). Returns same shape as x0.
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float, device=x0.device)
    t = t.to(x0.device)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    # alpha_t: probability of replacing each token with uniform noise
    alpha_t = 1.0 - torch.exp(-t * K / (K - 1))  # (B,) or (1,)
    if x0.dim() == 2:
        alpha_t = alpha_t.unsqueeze(-1)  # (B, 1) for broadcasting over L
    mask = torch.rand_like(x0.float()) < alpha_t
    noise = torch.randint(0, K, x0.shape, device=x0.device)
    return torch.where(mask, noise, x0)


class CTMCDenoiser(nn.Module):
    """Network that predicts clean-data logits from noisy sequence + time."""
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
        # x_t: (B, L) integers, t: (B,) floats in [0, 1]
        h = self.embed(x_t)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)  # broadcast time to all positions
        h = self.net(h)
        logits = self.out(h)  # (B, L, K)
        return logits


def compute_rate_from_denoiser(
    model: CTMCDenoiser, x_t: torch.Tensor, t: torch.Tensor, K: int
) -> torch.Tensor:
    """Compute transition rates from denoiser predictions.
    Rates proportional to predicted clean-data probabilities for non-current states.
    Returns (B, L, K) where rate to current state is negative (row sums to 0).
    """
    logits = model(x_t, t)  # (B, L, K)
    probs = F.softmax(logits, dim=-1)  # (B, L, K)
    B, L, _ = probs.shape
    current = x_t.unsqueeze(-1)  # (B, L, 1)
    mask = torch.zeros_like(probs).scatter_(2, current, 1.0)
    rates = probs * (1.0 - mask)
    diag = -rates.sum(dim=-1, keepdim=True)
    rates = rates + mask * diag
    return rates
