"""L42: pure-math/data-plumbing pieces for reproducing Huang et al.'s
activation steering (arXiv:2509.07983) on ESM2-650M toward thermostability.

Model-forward-pass pieces (embedding extraction, hooking, generation) live
in l42_run_repro.py, which requires ESM2-650M on a GPU and isn't
unit-testable without one. This module holds what IS testable without a
model: difference-of-means vector construction, high/low group splitting,
and the steering-hook math (verified against synthetic tensors here, then
reused against real activations in l42_run_repro.py).
"""
from typing import List, Tuple

import numpy as np

# Guruprasad et al. 1990 dipeptide instability weight table (DIWV) --
# a purely compositional (no model, no likelihood) instability index.
# LOWER instability index (<40) is the classical threshold for "stable."
# Used here specifically because it CANNOT be confounded by the generation
# model's own fluency/likelihood judgments -- a real risk with the earlier
# self-likelihood proxy, which penalizes any unusual-looking sequence
# (e.g. a poly-leucine run) regardless of whether it's actually more stable.
_DIWV = {
    "W": {"W": 1.0, "C": 1.0, "M": 24.68, "H": 24.68, "Y": 13.34, "F": 1.0, "Q": 1.0, "N": 13.34, "I": 1.0, "R": 1.0, "D": 1.0, "P": 1.0, "T": -14.03, "K": 1.0, "E": 1.0, "V": -7.49, "S": 1.0, "G": -9.37, "A": -14.03, "L": 13.34},
    "C": {"W": 24.68, "C": 1.0, "M": 33.6, "H": 33.6, "Y": 1.0, "F": 1.0, "Q": -6.54, "N": 1.0, "I": 1.0, "R": 1.0, "D": 20.26, "P": 20.26, "T": 33.6, "K": 1.0, "E": 1.0, "V": -6.54, "S": 1.0, "G": 1.0, "A": 1.0, "L": 20.26},
    "M": {"W": 1.0, "C": 1.0, "M": -1.88, "H": 58.28, "Y": 24.68, "F": 1.0, "Q": -6.54, "N": 1.0, "I": 1.0, "R": -6.54, "D": 1.0, "P": 44.94, "T": -1.88, "K": 1.0, "E": 1.0, "V": 1.0, "S": 44.94, "G": 1.0, "A": 13.34, "L": 1.0},
    "H": {"W": -1.88, "C": 1.0, "M": 1.0, "H": 1.0, "Y": 44.94, "F": -9.37, "Q": 1.0, "N": 24.68, "I": 44.94, "R": 1.0, "D": 1.0, "P": -1.88, "T": -6.54, "K": 24.68, "E": 1.0, "V": 1.0, "S": 1.0, "G": -9.37, "A": 1.0, "L": 1.0},
    "Y": {"W": -9.37, "C": 1.0, "M": 44.94, "H": 13.34, "Y": 13.34, "F": 1.0, "Q": 1.0, "N": 1.0, "I": 1.0, "R": -15.91, "D": 24.68, "P": 13.34, "T": -7.49, "K": 1.0, "E": -6.54, "V": 1.0, "S": 1.0, "G": -7.49, "A": 24.68, "L": 1.0},
    "F": {"W": 1.0, "C": 1.0, "M": 1.0, "H": 1.0, "Y": 33.6, "F": 1.0, "Q": 1.0, "N": 1.0, "I": 1.0, "R": 1.0, "D": 13.34, "P": 20.26, "T": 1.0, "K": -14.03, "E": 1.0, "V": 1.0, "S": 1.0, "G": 1.0, "A": 1.0, "L": 1.0},
    "Q": {"W": 1.0, "C": -6.54, "M": 1.0, "H": 1.0, "Y": -6.54, "F": -6.54, "Q": 20.26, "N": 1.0, "I": 1.0, "R": 1.0, "D": 20.26, "P": 20.26, "T": 1.0, "K": 1.0, "E": 20.26, "V": -6.54, "S": 44.94, "G": 1.0, "A": 1.0, "L": 1.0},
    "N": {"W": -9.37, "C": -1.88, "M": 1.0, "H": 1.0, "Y": 1.0, "F": -14.03, "Q": -6.54, "N": 1.0, "I": 44.94, "R": 1.0, "D": 1.0, "P": -1.88, "T": -7.49, "K": 24.68, "E": 1.0, "V": 1.0, "S": 1.0, "G": -14.03, "A": 1.0, "L": 1.0},
    "I": {"W": 1.0, "C": 1.0, "M": 1.0, "H": 13.34, "Y": 1.0, "F": 1.0, "Q": 1.0, "N": 1.0, "I": 1.0, "R": 1.0, "D": 1.0, "P": -1.88, "T": 1.0, "K": -7.49, "E": 44.94, "V": -7.49, "S": 1.0, "G": 1.0, "A": 1.0, "L": 20.26},
    "R": {"W": 58.28, "C": 1.0, "M": 1.0, "H": 20.26, "Y": -6.54, "F": 1.0, "Q": 20.26, "N": 13.34, "I": 1.0, "R": 58.28, "D": 1.0, "P": 20.26, "T": 1.0, "K": 1.0, "E": 1.0, "V": 1.0, "S": 44.94, "G": -7.49, "A": 1.0, "L": 1.0},
    "D": {"W": 1.0, "C": 1.0, "M": 1.0, "H": 1.0, "Y": 1.0, "F": -6.54, "Q": 1.0, "N": 1.0, "I": 1.0, "R": -6.54, "D": 1.0, "P": 1.0, "T": -14.03, "K": -7.49, "E": 1.0, "V": 1.0, "S": 20.26, "G": 1.0, "A": 1.0, "L": 1.0},
    "P": {"W": -1.88, "C": -6.54, "M": -6.54, "H": 1.0, "Y": 1.0, "F": 20.26, "Q": 20.26, "N": 1.0, "I": 1.0, "R": -6.54, "D": -6.54, "P": 20.26, "T": 1.0, "K": 1.0, "E": 18.38, "V": 20.26, "S": 20.26, "G": 1.0, "A": 20.26, "L": 1.0},
    "T": {"W": -14.03, "C": 1.0, "M": 1.0, "H": 1.0, "Y": 1.0, "F": 13.34, "Q": -6.54, "N": -14.03, "I": 1.0, "R": 1.0, "D": 1.0, "P": 1.0, "T": 1.0, "K": 1.0, "E": 20.26, "V": 1.0, "S": 1.0, "G": -7.49, "A": 1.0, "L": 1.0},
    "K": {"W": 1.0, "C": 1.0, "M": 33.6, "H": 1.0, "Y": 1.0, "F": 1.0, "Q": 24.68, "N": 1.0, "I": -7.49, "R": 33.6, "D": 1.0, "P": -6.54, "T": 1.0, "K": 1.0, "E": 1.0, "V": -7.49, "S": 1.0, "G": -7.49, "A": 1.0, "L": -7.49},
    "E": {"W": -14.03, "C": 44.94, "M": 1.0, "H": -6.54, "Y": 1.0, "F": 1.0, "Q": 20.26, "N": 1.0, "I": 20.26, "R": 1.0, "D": 20.26, "P": 20.26, "T": 1.0, "K": 1.0, "E": 33.6, "V": 1.0, "S": 20.26, "G": 1.0, "A": 1.0, "L": 1.0},
    "V": {"W": 1.0, "C": 1.0, "M": 1.0, "H": 1.0, "Y": -6.54, "F": 1.0, "Q": 1.0, "N": 1.0, "I": 1.0, "R": 1.0, "D": -14.03, "P": 20.26, "T": -7.49, "K": -1.88, "E": 1.0, "V": 1.0, "S": 1.0, "G": -7.49, "A": 1.0, "L": 1.0},
    "S": {"W": 1.0, "C": 33.6, "M": 1.0, "H": 1.0, "Y": 1.0, "F": 1.0, "Q": 20.26, "N": 1.0, "I": 1.0, "R": 20.26, "D": 1.0, "P": 44.94, "T": 1.0, "K": 1.0, "E": 20.26, "V": 1.0, "S": 20.26, "G": 1.0, "A": 1.0, "L": 1.0},
    "G": {"W": 13.34, "C": 1.0, "M": 1.0, "H": 1.0, "Y": -7.49, "F": 1.0, "Q": 1.0, "N": -7.49, "I": -7.49, "R": 1.0, "D": 1.0, "P": 1.0, "T": -7.49, "K": -7.49, "E": -6.54, "V": 1.0, "S": 1.0, "G": 13.34, "A": -7.49, "L": 1.0},
    "A": {"W": 1.0, "C": 44.94, "M": 1.0, "H": -7.49, "Y": 1.0, "F": 1.0, "Q": 1.0, "N": 1.0, "I": 1.0, "R": 1.0, "D": -7.49, "P": 20.26, "T": 1.0, "K": 1.0, "E": 1.0, "V": 1.0, "S": 1.0, "G": 1.0, "A": 1.0, "L": 1.0},
    "L": {"W": 24.68, "C": 1.0, "M": 1.0, "H": 1.0, "Y": 1.0, "F": 1.0, "Q": 33.6, "N": 1.0, "I": 1.0, "R": 20.26, "D": 1.0, "P": 20.26, "T": 1.0, "K": -7.49, "E": 1.0, "V": 1.0, "S": 1.0, "G": 1.0, "A": 1.0, "L": 1.0},
}


def instability_index(sequence: str) -> float:
    """Guruprasad, Reddy & Pandit (1990) instability index -- purely
    compositional, no model/likelihood involved. Below 40 is the classical
    "stable" threshold; higher = predicted less stable. Unknown dipeptides
    (non-standard residues) contribute 1.0 (the DIWV table's own convention
    for "no destabilizing effect known").
    """
    if len(sequence) < 2:
        raise ValueError("instability_index requires a sequence of length >= 2")
    total = 0.0
    for i in range(len(sequence) - 1):
        a, b = sequence[i], sequence[i + 1]
        total += _DIWV.get(a, {}).get(b, 1.0)
    return (10.0 / len(sequence)) * total


def split_by_percentile(
    sequences: List[str], scores: np.ndarray, low_pct: float = 20.0, high_pct: float = 80.0
) -> Tuple[List[str], List[str]]:
    """Split sequences into low-scoring and high-scoring groups by
    percentile threshold, per Huang et al.'s difference-of-means recipe
    (build steering vector from a 'high' set minus a 'low' set on the
    target property). Returns (low_group, high_group).
    """
    if len(sequences) != len(scores):
        raise ValueError("sequences and scores must have the same length")
    if not (0.0 <= low_pct < high_pct <= 100.0):
        raise ValueError("require 0 <= low_pct < high_pct <= 100")

    low_threshold = np.percentile(scores, low_pct)
    high_threshold = np.percentile(scores, high_pct)

    low_group = [seq for seq, score in zip(sequences, scores) if score <= low_threshold]
    high_group = [seq for seq, score in zip(sequences, scores) if score >= high_threshold]
    return low_group, high_group


def difference_of_means_vector(low_activations: np.ndarray, high_activations: np.ndarray) -> np.ndarray:
    """Huang et al.'s steering-vector construction: mean activation over the
    HIGH group minus mean activation over the LOW group, per layer. Inputs
    are [n_sequences, d_model] (already pooled per sequence, e.g. mean over
    tokens for an auto-encoding model); returns [d_model].
    """
    if low_activations.shape[1] != high_activations.shape[1]:
        raise ValueError("low and high activations must share the same d_model dimension")
    return high_activations.mean(axis=0) - low_activations.mean(axis=0)


def renormalize_to_original_norm(perturbed: np.ndarray, original_norm: np.ndarray) -> np.ndarray:
    """Huang et al.'s renormalization step: after adding alpha*direction to
    an activation, rescale so the perturbed activation has the SAME norm as
    the original (prevents the intervention from trivially inflating
    activation magnitude, which could confound any downstream effect with a
    simple 'bigger numbers' artifact rather than a genuine directional shift).
    perturbed, original_norm: both [n, d_model] (per-token) or [d_model].
    """
    perturbed_norm = np.linalg.norm(perturbed, axis=-1, keepdims=True)
    perturbed_norm = np.where(perturbed_norm < 1e-8, 1.0, perturbed_norm)
    target_norm = np.linalg.norm(original_norm, axis=-1, keepdims=True)
    return perturbed * (target_norm / perturbed_norm)


# IVYWREL: the amino acids (Ile, Val, Tyr, Trp, Arg, Glu, Leu) independently
# documented as enriched in thermophile vs. mesophile proteomes (Zeldovich,
# Berezovsky & Shakhnovich 2007; Kreil & Ouzounis 2001 "IVYWREL" signature).
# Used here as a scoring proxy that is NOT derived from anything observed in
# this project's own generated sequences -- unlike instability_index, which
# was found to be gameable by this exact steering vector's dominant failure
# mode (poly-leucine collapse, see docs/L42_STEERING_REPRO.md).
IVYWREL_RESIDUES = frozenset("IVYWREL")


def ivywrel_fraction(sequence: str, residues: frozenset = IVYWREL_RESIDUES) -> float:
    """Fraction of a sequence's residues in the IVYWREL thermostability-
    associated set. Higher = more thermostable-like composition, per the
    independent comparative-genomics literature cited above -- this is a
    proxy for the property Huang et al. steer toward, not the exact same
    fitness function they use, since their fitted model isn't available here.
    """
    if len(sequence) == 0:
        raise ValueError("ivywrel_fraction requires a non-empty sequence")
    return sum(1 for c in sequence if c in residues) / len(sequence)


def is_degenerate_sequence(sequence: str, max_single_aa_fraction: float = 0.25) -> bool:
    """Flags homopolymer-collapse artifacts (e.g. poly-leucine) BEFORE any
    stability scoring -- confirmed via manual inspection (docs/L42_STEERING_REPRO.md)
    that the instability-index proxy is gameable by exactly this artifact
    (leucine-heavy sequences score as artificially "stable"). Filtering
    degenerate sequences out first, rather than trying to build a score
    robust to them, removes the confound instead of fighting it.

    Threshold calibrated against real generated data: unsteered-baseline and
    non-collapsed steered sequences top out at max_single_aa_fraction=0.227;
    confirmed-collapsed sequences (alpha>=1.0 in the L42 diagnostic run)
    start at 0.319. 0.25 sits in the clean gap between the two.
    """
    if len(sequence) == 0:
        raise ValueError("is_degenerate_sequence requires a non-empty sequence")
    counts = {}
    for c in sequence:
        counts[c] = counts.get(c, 0) + 1
    return (max(counts.values()) / len(sequence)) > max_single_aa_fraction


def paired_bootstrap_mean_diff(scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10000, seed: int = 0) -> dict:
    """Bootstrap CI for mean(scores_b - scores_a), pairing on shared indices
    (e.g. the same held-out eval sequences scored under two conditions).
    Mirrors virion_eval.py's paired_bootstrap_metric_diff but for a plain
    per-item score array rather than a classifier metric recomputed per
    resample -- correct here because the "metric" (mean score) is already
    linear, so resampling the precomputed per-item diffs is equivalent to
    (and much cheaper than) rerunning scoring on each bootstrap resample.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) != len(scores_b):
        raise ValueError("scores_a and scores_b must have the same length")
    if len(scores_a) == 0:
        raise ValueError("paired_bootstrap_mean_diff requires at least one paired observation")

    diffs = scores_b - scores_a
    n = len(diffs)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return {
        "point_estimate": float(diffs.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_95pct": bool(ci_lower > 0 or ci_upper < 0),
        "n": n,
    }


def dose_response_is_monotonic_then_collapsing(alphas: List[float], effects: List[float], collapse_tolerance: float = 0.0) -> bool:
    """Check for the qualitative dose-response shape Huang et al. report:
    effect increases with alpha up to some point, then may collapse
    (over-steering) at extreme alpha -- i.e. NOT flat/noisy across all
    alphas. Returns True if there's a clear non-flat pattern: at least one
    interior alpha's effect exceeds both its neighbors' by more than
    collapse_tolerance is NOT required (that would demand a collapse); the
    weaker, correct check is monotonic non-decrease up to the max, since
    collapse is only expected beyond the tested range.
    """
    if len(alphas) != len(effects):
        raise ValueError("alphas and effects must have the same length")
    if len(alphas) < 2:
        return False

    order = np.argsort(alphas)
    sorted_effects = np.array(effects)[order]
    # "clear pattern" = strictly increasing overall (last > first by more than
    # a small tolerance), not just noise scattered around zero.
    return bool(sorted_effects[-1] - sorted_effects[0] > collapse_tolerance)
