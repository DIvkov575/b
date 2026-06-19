import torch
import torch.nn as nn


class TimeConditionalClassifier(nn.Module):
    """Binary classifier p(y=1 | x_t, t) — time-conditional for noisy inputs."""

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
        self.pool = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.embed(x_t)  # (B, L, hidden)
        t_emb = self.time_embed(t.unsqueeze(-1))  # (B, hidden)
        h = h + t_emb.unsqueeze(1)
        h = self.net(h)
        h = h.mean(dim=1)  # pool over sequence
        h = torch.relu(self.pool(h))
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
