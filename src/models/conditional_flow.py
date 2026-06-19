import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalProbPathFlow(nn.Module):
    """Conditional discrete flow with CFG dropout.

    Trained to predict p(x_0 | x_t, t, c) where c is a multi-hot condition vector.
    At training time, c is randomly zeroed with probability cfg_dropout_prob
    to learn the unconditional p(x_0 | x_t, t) jointly.
    """

    def __init__(self, K: int, L: int, n_conditions: int,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.K = K
        self.L = L
        self.n_conditions = n_conditions

        self.embed = nn.Embedding(K, hidden_dim)
        self.pos_embed = nn.Embedding(L, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(n_conditions, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, K)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                cfg_dropout_prob: float = 0.0) -> torch.Tensor:
        """
        Args:
            x_t: (B, L) noisy token indices
            t: (B,) time values in [0, 1]
            cond: (B, n_conditions) multi-hot condition vector
            cfg_dropout_prob: probability of dropping condition (training only)
        Returns:
            (B, L, K) logits for p(x_0 | x_t, t, c)
        """
        B, L = x_t.shape

        # CFG dropout: zero the condition vector with probability cfg_dropout_prob
        if cfg_dropout_prob > 0.0 and self.training:
            drop_mask = (torch.rand(B, 1, device=x_t.device) < cfg_dropout_prob).float()
            cond = cond * (1.0 - drop_mask)

        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)
        t_emb = self.time_embed(t.unsqueeze(-1))
        c_emb = self.cond_embed(cond)
        h = h + t_emb.unsqueeze(1) + c_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.out(h)
