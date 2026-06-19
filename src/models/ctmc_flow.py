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
    """Network that predicts clean-data logits from noisy sequence + time.
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
        h = self.embed(x_t) + self.pos_embed(pos_ids)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)  # (B, L, hidden)
        return self.out(h)  # (B, L, K)


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
