"""ProteinBERT MLM model. Ported unchanged from PFold (github.com/DIvkov575/PFold,
commit f44eecc) model.py."""
import math

import torch
import torch.nn as nn


class ProteinBERT(nn.Module):
    def __init__(self, vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, max_length: int = 512, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlm_head = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)

        if attention_mask is not None:
            attention_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)

        logits = self.mlm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return {'loss': loss, 'logits': logits, 'hidden_states': x}


def create_model(vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                  n_heads: int = 8, d_ff: int = 1024, max_length: int = 512,
                  dropout: float = 0.1) -> ProteinBERT:
    model = ProteinBERT(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_length=max_length, dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model w/ {total_params:,} total parameters")
    return model
