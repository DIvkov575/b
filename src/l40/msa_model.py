"""Top-level MSA-aware MLM model: query token embedding + sinusoidal position
encoding, optionally enriched by (a) Boltz's real profile/deletion_mean
InputEmbedder branch and (b) the MSAModule cross-homolog embedding (Task 3's
msa_module.py, Pairformer removed) -- then a standard transformer encoder
trunk (same shape as ProteinBERT) + MLM head. Each enrichment is an
independent on/off switch, matching the ablation's ordinary arms.
"""
import math

import torch
import torch.nn as nn

from src.l40.msa_module import MSAModule


class MSAAwareProteinBERT(nn.Module):
    def __init__(self, vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, max_length: int = 512, dropout: float = 0.1,
                 msa_s: int = 64, token_z: int = 32, msa_blocks: int = 2,
                 use_deletion_features: bool = True, use_profile: bool = True,
                 use_msa_module: bool = True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_deletion_features = use_deletion_features
        self.use_profile = use_profile
        self.use_msa_module = use_msa_module

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)

        if use_profile:
            # Matches Boltz's real InputEmbedder branch: [profile, deletion_mean]
            # concatenated and linearly projected, added into the single repr.
            self.profile_proj = nn.Linear(vocab_size + 1, d_model, bias=False)

        if use_msa_module:
            self.msa_module = MSAModule(
                msa_s=msa_s, token_z=token_z, token_s=d_model, msa_blocks=msa_blocks,
                vocab_size=vocab_size,
            )
            self.msa_out_proj = nn.Linear(msa_s, d_model, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
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

    def forward(self, msa_tokens: torch.Tensor, has_deletion: torch.Tensor,
                deletion_value: torch.Tensor, profile: torch.Tensor,
                deletion_mean: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None):
        # msa_tokens: (B, S, N) int ids, row 0 of S is the query (masked) sequence.
        query_ids = msa_tokens[:, 0]
        seq_len = query_ids.size(1)

        x = self.embedding(query_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]

        if self.use_profile:
            profile_feat = torch.cat([profile, deletion_mean.unsqueeze(-1)], dim=-1)
            x = x + self.profile_proj(profile_feat)

        if self.use_msa_module:
            B, S, N = msa_tokens.shape
            msa_onehot = torch.nn.functional.one_hot(msa_tokens, self.vocab_size).float()
            if not self.use_deletion_features:
                has_deletion = torch.zeros_like(has_deletion)
                deletion_value = torch.zeros_like(deletion_value)
            msa_mask = torch.ones(B, S, N, device=msa_tokens.device)
            if attention_mask is not None:
                token_pair_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            else:
                token_pair_mask = torch.ones(B, N, N, device=msa_tokens.device)

            msa_query_repr = self.msa_module(
                msa_onehot, has_deletion, deletion_value, msa_mask, token_pair_mask, x,
            )
            x = x + self.msa_out_proj(msa_query_repr)

        x = self.dropout(x)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        logits = self.mlm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return {'loss': loss, 'logits': logits, 'hidden_states': x}
