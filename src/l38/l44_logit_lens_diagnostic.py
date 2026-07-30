"""L44: test whether L42/L43's over-steering collapse residue (leucine for
thermostability, alanine/glycine for solubility) is predictable in advance
from the steering vector itself, via a logit-lens-style projection through
the model's own unembedding (lm_head), BEFORE ever generating a single
sequence.

Motivation: in both L42 and L43, pushing alpha too high didn't produce
random garbage -- it collapsed to one specific, consistent amino acid (or
pair), and that residue was the one over-represented in whichever group
built the vector. This script checks whether that collapse residue is
visible directly in the vector's projection onto the vocabulary, which
would mean over-steering is not an arbitrary failure mode but a predictable
consequence of pushing the residual stream toward whatever token the
model's own unembedding matrix associates most strongly with that
direction -- checkable cheaply, without any generation, for a NEW target
property before ever running the expensive alpha sweep.

Reuses L42/L43's exact vector-construction code and data (same seed) so
the vectors checked here are identical to the ones that actually steered
generation in those runs -- this is a diagnostic ON the existing results,
not a new experiment.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_steering_repro import difference_of_means_vector, split_by_percentile

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
MAX_SEQ_LEN = 400
N_VECTOR_SEQS_PER_GROUP = 150
SEED = 0

MELTOME_PATH = Path(__file__).resolve().parent / "data_cache" / "meltome" / "mixed_split.csv"
SOLUBILITY_PATH = Path(__file__).resolve().parent / "data_cache" / "solubility" / "train.csv"
OUT_PATH = Path(__file__).resolve().parent / "l44_logit_lens_out.json"


@torch.no_grad()
def mean_pooled_activation_all_layers(model, tokenizer, sequences, device, max_len=MAX_SEQ_LEN):
    n_layers = model.config.num_hidden_layers
    per_layer_activations = {layer: [] for layer in range(n_layers)}
    for seq in sequences:
        seq = seq[:max_len]
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
        out = model(**enc, output_hidden_states=True)
        for layer in range(n_layers):
            hidden = out.hidden_states[layer + 1].squeeze(0).float()
            per_layer_activations[layer].append(hidden.mean(dim=0).cpu().numpy())
    return {layer: np.stack(vals, axis=0) for layer, vals in per_layer_activations.items()}


def build_steering_vectors(model, tokenizer, device, low_group, high_group):
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    high_activations = mean_pooled_activation_all_layers(model, tokenizer, high_group, device)
    n_layers = model.config.num_hidden_layers
    vectors = {}
    for layer in range(n_layers):
        vec = difference_of_means_vector(low_activations[layer], high_activations[layer])
        vectors[layer] = torch.tensor(vec, dtype=torch.float32, device=device)
    return vectors


STANDARD_AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


@torch.no_grad()
def logit_lens_top_tokens(model, tokenizer, vector, top_k=5, restrict_to_standard_aa=True):
    """Project a residual-stream direction onto the model's (tied)
    input/output embedding matrix via COSINE similarity, and return the
    top_k highest-scoring vocabulary tokens.

    Two things were checked and fixed before trusting this:
    1. NOT routed through the full lm_head (dense -> layer_norm ->
       decoder): a self-identity smoke test showed that doing so does NOT
       recover a token's own embedding as its own top match (leucine's
       embedding projects to K/M/<unk> through the full head, since that
       transform is fit for real hidden states, not an arbitrary
       difference vector fed in directly).
    2. NOT raw dot product: non-standard/rare amino-acid codes (B, U, Z,
       O, X) have outlier embedding NORMS (up to 6.15 vs. ~2.4-3.4 for the
       20 standard residues, confirmed by direct inspection), which
       dominated raw dot-product rankings regardless of direction and
       produced a nonsensical top-1 (e.g. "B") on both L42 and L43's real
       vectors. Cosine similarity removes the norm confound. Restricting
       to the 20 standard amino acids (restrict_to_standard_aa=True,
       default) additionally removes rare/special tokens from
       consideration entirely, since only standard residues were ever
       observed in the actual collapse.
    """
    emb_matrix = model.esm.embeddings.word_embeddings.weight
    cos = torch.nn.functional.cosine_similarity(emb_matrix, vector.unsqueeze(0), dim=-1)

    if restrict_to_standard_aa:
        allowed_ids = {tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AMINO_ACIDS}
        mask = torch.full_like(cos, float("-inf"))
        for tid in allowed_ids:
            mask[tid] = 0.0
        cos = cos + mask

    top_scores, top_ids = torch.topk(cos, top_k)
    tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    return list(zip(tokens, top_scores.tolist()))


def load_thermostability_groups():
    df = pd.read_csv(MELTOME_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    low_group, high_group = split_by_percentile(
        vector_pool["sequence"].tolist(), vector_pool["label"].values, low_pct=20.0, high_pct=80.0
    )
    return low_group[:N_VECTOR_SEQS_PER_GROUP], high_group[:N_VECTOR_SEQS_PER_GROUP]


def load_solubility_groups():
    df = pd.read_csv(SOLUBILITY_PATH)
    df = df[df["sequences"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    insoluble_pool = vector_pool[vector_pool["labels"] == 0]["sequences"].tolist()
    soluble_pool = vector_pool[vector_pool["labels"] == 1]["sequences"].tolist()
    return insoluble_pool[:N_VECTOR_SEQS_PER_GROUP], soluble_pool[:N_VECTOR_SEQS_PER_GROUP]


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers

    results = {}
    for target_name, loader, observed_collapse_residues in [
        ("thermostability", load_thermostability_groups, ["L"]),
        ("solubility", load_solubility_groups, ["A", "G"]),
    ]:
        print(f"\n=== {target_name} ===", flush=True)
        low_group, high_group = loader()
        vectors = build_steering_vectors(model, tokenizer, device, low_group, high_group)

        per_layer_top = {}
        for layer in range(n_layers):
            top = logit_lens_top_tokens(model, tokenizer, vectors[layer], top_k=5)
            per_layer_top[layer] = top

        # Aggregate: which token is #1 most often across layers, and does
        # it match the empirically observed collapse residue(s)?
        top1_counts = {}
        for layer, top in per_layer_top.items():
            token = top[0][0]
            top1_counts[token] = top1_counts.get(token, 0) + 1
        top1_sorted = sorted(top1_counts.items(), key=lambda kv: -kv[1])

        matches_observed = any(tok in observed_collapse_residues for tok, _ in top1_sorted[:3])

        print(f"top-1 token by layer (aggregated): {top1_sorted[:5]}", flush=True)
        print(f"observed collapse residue(s) from generation: {observed_collapse_residues}", flush=True)
        print(f"logit-lens predicts this collapse: {matches_observed}", flush=True)

        results[target_name] = {
            "per_layer_top5": {str(l): t for l, t in per_layer_top.items()},
            "top1_aggregated": top1_sorted,
            "observed_collapse_residues": observed_collapse_residues,
            "logit_lens_predicts_collapse": matches_observed,
        }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
