import torch
import torch.nn as nn
import torch.nn.functional as F


class MDLM(nn.Module):
    """Masked Diffusion Language Model.
    Predicts p(x_0 | x_t, t) at each position.
    Architecture: Transformer encoder with time conditioning.
    """

    def __init__(self, vocab_size: int = 28, seq_len: int = 256,
                 hidden_dim: int = 256, num_layers: int = 6, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(seq_len, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, L, V)."""
        B, L = x_t.shape
        pos_ids = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.embed(x_t) + self.pos_embed(pos_ids)
        t_emb = self.time_embed(t.unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)
        h = self.transformer(h)
        return self.out(h)

    def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns log p(x_0 | x_t, t) — shape (B, L, V)."""
        logits = self.forward(x_t, t)
        return F.log_softmax(logits, dim=-1)
