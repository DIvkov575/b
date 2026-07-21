"""L41: causal activation steering with ESM-C SAE function-directions.

Pure-math / data-plumbing pieces factored out for testability (no GPU/model
required); the model-forward-pass pieces (embedding extraction, hooking,
mask-fill generation) live in l41_run_gate1.py / l41_run_gate2.py, which
require ESMC-300M loaded on a GPU and are not unit-testable without one.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    """Standardized mean difference between two 1D samples.

    Uses pooled standard deviation (sample, ddof=1). Returns 0.0 if pooled
    std is exactly zero (degenerate constant-valued arrays) to avoid NaN/Inf
    propagating into downstream feature ranking.
    """
    n1, n2 = len(positive), len(negative)
    if n1 < 2 or n2 < 2:
        raise ValueError("cohens_d requires at least 2 samples per group")

    mean_diff = positive.mean() - negative.mean()
    pooled_var = ((n1 - 1) * positive.var(ddof=1) + (n2 - 1) * negative.var(ddof=1)) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0.0:
        return 0.0
    return float(mean_diff / pooled_std)


def rank_features_by_separation(
    positive_features: np.ndarray, negative_features: np.ndarray
) -> List[Tuple[int, float]]:
    """Given [n_pos, n_features] and [n_neg, n_features] sparse-feature
    activation matrices (e.g. mean-pooled per-sequence SAE codes), return
    (feature_index, cohens_d) sorted by |cohens_d| descending."""
    if positive_features.shape[1] != negative_features.shape[1]:
        raise ValueError("positive and negative feature matrices must have the same feature dimension")

    n_features = positive_features.shape[1]
    results = []
    for i in range(n_features):
        d = cohens_d(positive_features[:, i], negative_features[:, i])
        results.append((i, d))
    results.sort(key=lambda pair: abs(pair[1]), reverse=True)
    return results


def gate1_decision(ranked_features: List[Tuple[int, float]], threshold: float = 1.0) -> Dict:
    """Apply the L41 Gate 1 pre-registered PASS/KILL rule (docs/L41_PROTOCOL.md):
    PASS iff the top-ranked feature's |Cohen's d| exceeds `threshold`."""
    if not ranked_features:
        return {"decision": "KILL", "reason": "no features to rank", "winning_feature": None, "effect_size": None}

    winning_idx, winning_d = ranked_features[0]
    if abs(winning_d) > threshold:
        return {
            "decision": "PASS",
            "reason": f"top feature {winning_idx} has |d|={abs(winning_d):.3f} > {threshold}",
            "winning_feature": winning_idx,
            "effect_size": winning_d,
        }
    return {
        "decision": "KILL",
        "reason": f"best |d|={abs(winning_d):.3f} does not exceed threshold {threshold}",
        "winning_feature": None,
        "effect_size": winning_d,
    }


def fit_zscore_stats(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-dimension mean/std from a [n, d_model] sample of activations.
    Per the ESM-C SAE paper's methodology ("Each input to the encoder is
    Z-score normalized"), this must be applied before sae_encode -- omitting
    it changes which feature wins a Cohen's-d search (verified empirically:
    feature 7196 wins unnormalized, feature 10004 wins normalized, on the
    same kinase/non-kinase data -- see docs/L41_PROTOCOL.md's post-hoc
    correction section). Returns (mean, std) with std floored at 1e-6 to
    avoid divide-by-zero on a degenerate constant dimension.
    """
    mean = activations.mean(axis=0)
    std = activations.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def zscore_normalize(activation: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (activation - mean) / std


def sae_encode(activation: np.ndarray, W_enc: np.ndarray, b_dec: np.ndarray, k: int) -> np.ndarray:
    """Standard top-k sparse-autoencoder encode: pre_act = (x - b_dec) @ W_enc,
    keep only the top-k values (ReLU'd), zero elsewhere. activation may be
    [d_model] or [n, d_model]; returns matching leading shape with codebook_dim
    as the last axis.

    IMPORTANT: `activation` must already be Z-score normalized (see
    fit_zscore_stats/zscore_normalize above) before calling this -- the SAE
    was trained on normalized inputs, and skipping this step silently picks
    a different, wrong "winning" feature in a Cohen's-d search (see
    docs/L41_PROTOCOL.md's post-hoc correction section for the empirical
    before/after).
    """
    pre_act = (activation - b_dec) @ W_enc
    single = pre_act.ndim == 1
    if single:
        pre_act = pre_act[None, :]

    codebook_dim = pre_act.shape[-1]
    k = min(k, codebook_dim)
    topk_idx = np.argpartition(-pre_act, k - 1, axis=-1)[:, :k]
    sparse = np.zeros_like(pre_act)
    rows = np.arange(pre_act.shape[0])[:, None]
    vals = np.take_along_axis(pre_act, topk_idx, axis=-1)
    sparse[rows, topk_idx] = np.maximum(vals, 0.0)

    return sparse[0] if single else sparse


def sae_decode(features: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    """features @ W_dec + b_dec."""
    return features @ W_dec + b_dec


def load_sequences_with_labels(
    positive_fasta: Path, negative_fasta: Path, parse_fn
) -> Tuple[List[str], np.ndarray]:
    """Thin wrapper combining two label-separated FASTA files into
    (sequences, labels) with labels 1=positive, 0=negative. parse_fn is
    injected (expected: src.l38.phage_data.parse_fasta) to avoid a hard
    import-time dependency for pure-math unit tests."""
    positive = parse_fn(positive_fasta)
    negative = parse_fn(negative_fasta)
    sequences = positive + negative
    labels = np.array([1] * len(positive) + [0] * len(negative))
    return sequences, labels
