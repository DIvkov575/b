import torch
import torch.nn as nn


class TimeConditionalClassifier(nn.Module):
    """Binary classifier p(y=1 | x_t, t) — time-conditional for noisy inputs.
    Uses Transformer for inter-position communication (needed for pattern detection).
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
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x_t.shape
        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)  # (B, L, hidden)
        h = h.mean(dim=1)  # pool over sequence
        return self.head(h)  # (B, 1)

    def predict_prob(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x_t, t).squeeze(-1)
        return torch.sigmoid(logits)

    def log_prob_ratio(
        self, x: torch.Tensor, x_prime: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute log[p(y|x')/p(y|x)] — the guidance signal for CTMC rates."""
        log_p_x = torch.log(self.predict_prob(x, t).clamp(min=1e-8))
        log_p_x_prime = torch.log(self.predict_prob(x_prime, t).clamp(min=1e-8))
        return log_p_x_prime - log_p_x
