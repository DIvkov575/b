"""MSA-aware embedding stack, ported from Boltz's real trunk (verified against
boltz.model.modules.trunkv2, boltz.model.layers.pair_averaging,
boltz.model.layers.outer_product_mean, boltz.model.layers.transition) with the
structure-specific PairformerNoSeqLayer call removed from MSALayer — nothing
downstream of this module needs a structure/distance prediction, only an MLM
head over the query sequence. Everything else (MSA embedding, cross-sequence
attention via PairWeightedAveraging, co-evolution signal via OuterProductMean,
the per-sequence Transition) is kept as in the real Boltz trunk.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transition(nn.Module):
    """SwiGLU-gated MLP. dim -> hidden -> dim, LayerNorm first."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.silu(self.fc1(x)) * self.fc2(x)
        return self.fc3(x)


class OuterProductMean(nn.Module):
    """Builds a pair representation z (B,N,N,c_out) from an MSA representation
    m (B,S,N,c_in) via an outer product over the hidden projection, averaged
    across the MSA-sequence axis S. This is where co-evolution signal (two
    positions that covary across homologs) enters the pair tensor."""

    def __init__(self, c_in: int, c_hidden: int, c_out: int):
        super().__init__()
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_in)
        self.proj_a = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_b = nn.Linear(c_in, c_hidden, bias=False)
        self.proj_o = nn.Linear(c_hidden * c_hidden, c_out)

    def forward(self, m: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1).to(m)  # (B, S, N, 1)
        m = self.norm(m)
        a = self.proj_a(m) * mask  # (B, S, N, c_hidden)
        b = self.proj_b(m) * mask  # (B, S, N, c_hidden)

        num_mask = (mask[:, :, None, :, 0] * mask[:, :, :, None, 0]).sum(dim=1)
        num_mask = num_mask.unsqueeze(-1).clamp(min=1)  # (B, N, N, 1)

        z = torch.einsum("bsic,bsjd->bijcd", a, b)  # (B, N, N, c_hidden, c_hidden)
        z = z.reshape(*z.shape[:3], -1)  # (B, N, N, c_hidden*c_hidden)
        z = z / num_mask
        return self.proj_o(z)


class PairWeightedAveraging(nn.Module):
    """Aggregates the MSA representation m across the MSA-sequence axis S,
    with attention weights derived from the pair tensor z (not from m itself)
    — this is the module's real cross-sequence mixing mechanism."""

    def __init__(self, c_m: int, c_z: int, c_h: int, num_heads: int, inf: float = 1e6):
        super().__init__()
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)

    def forward(self, m: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # m: (B, S, N, c_m), z: (B, N, N, c_z), mask: (B, N, N)
        B, S, N, _ = m.shape
        m = self.norm_m(m)
        z = self.norm_z(z)

        v = self.proj_m(m).view(B, S, N, self.num_heads, self.c_h)
        v = v.permute(0, 3, 1, 2, 4)  # (B, H, S, N, c_h)

        b = self.proj_z(z).permute(0, 3, 1, 2)  # (B, H, N, N)
        b = b + (1 - mask[:, None]) * -self.inf
        w = torch.softmax(b, dim=-1)  # (B, H, N, N)

        g = torch.sigmoid(self.proj_g(m)).view(B, S, N, self.num_heads, self.c_h)
        g = g.permute(0, 3, 1, 2, 4)  # (B, H, S, N, c_h)

        o = torch.einsum("bhij,bhsjd->bhsid", w, v)  # (B, H, S, N, c_h)
        o = o.permute(0, 2, 3, 1, 4).reshape(B, S, N, self.num_heads * self.c_h)
        g = g.permute(0, 2, 3, 1, 4).reshape(B, S, N, self.num_heads * self.c_h)

        return self.proj_o(g * o)


class MSALayer(nn.Module):
    """One MSA-processing block: cross-sequence mixing (PairWeightedAveraging)
    + per-sequence transition, updating m; then OuterProductMean updates z from
    the new m. The real Boltz MSALayer also runs a PairformerNoSeqLayer on z
    here — intentionally omitted, since this ablation has no structure/distance
    prediction downstream that needs a refined pair representation."""

    def __init__(self, msa_s: int, token_z: int, c_h: int = 32, num_heads: int = 8,
                 hidden_mult: int = 4):
        super().__init__()
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s, c_z=token_z, c_h=c_h, num_heads=num_heads,
        )
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * hidden_mult)
        self.outer_product_mean = OuterProductMean(c_in=msa_s, c_hidden=32, c_out=token_z)

    def forward(self, m: torch.Tensor, z: torch.Tensor, msa_mask: torch.Tensor,
                token_pair_mask: torch.Tensor):
        m = m + self.pair_weighted_averaging(m, z, token_pair_mask)
        m = m + self.msa_transition(m)
        z = z + self.outer_product_mean(m, msa_mask)
        return m, z


class MSAModule(nn.Module):
    """Embeds a batch of MSA rows (one-hot tokens + deletion features) into an
    MSA representation, broadcasts in the query's own single-sequence
    embedding, runs msa_blocks MSALayers, and returns the query row (index 0
    of the sequence axis) of the final MSA representation — this is what
    feeds the downstream MLM head."""

    def __init__(self, msa_s: int, token_z: int, token_s: int, msa_blocks: int,
                 vocab_size: int, c_h: int = 32, num_heads: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.s_proj = nn.Linear(token_s, msa_s, bias=False)
        self.msa_proj = nn.Linear(vocab_size + 2, msa_s, bias=False)
        self.layers = nn.ModuleList([
            MSALayer(msa_s=msa_s, token_z=token_z, c_h=c_h, num_heads=num_heads)
            for _ in range(msa_blocks)
        ])
        # Real Boltz's z arrives pre-populated from an upstream Pairformer
        # trunk (excluded here -- nothing downstream needs it). Without that,
        # z starts at zero and PairWeightedAveraging's attention weights (b =
        # proj_z(z)) carry no cross-homolog signal until OuterProductMean
        # updates z at the END of a block -- one block too late for that
        # signal to reach the query. This initial pass seeds z from the raw
        # MSA embedding before the first block runs, so even msa_blocks=1
        # lets homolog information reach the query representation.
        self.z_seed = OuterProductMean(c_in=msa_s, c_hidden=32, c_out=token_z)

    def forward(self, msa_onehot: torch.Tensor, has_deletion: torch.Tensor,
                deletion_value: torch.Tensor, msa_mask: torch.Tensor,
                token_pair_mask: torch.Tensor, single_emb: torch.Tensor) -> torch.Tensor:
        # msa_onehot: (B, S, N, vocab_size); has_deletion/deletion_value: (B, S, N)
        m = torch.cat([msa_onehot, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)], dim=-1)
        m = self.msa_proj(m)
        m = m + self.s_proj(single_emb).unsqueeze(1)

        z = self.z_seed(m, msa_mask)

        for layer in self.layers:
            m, z = layer(m, z, msa_mask, token_pair_mask)

        return m[:, 0]  # query row
