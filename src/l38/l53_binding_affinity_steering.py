"""L53: extend the L42 activation-steering harness (validated on
thermostability, docs/L42_STEERING_REPRO.md) to a NEW target property --
binding affinity -- using a proxy validated against real experimental labels
BEFORE the run, per L50 criterion 4 (the L43 GRAVY lesson: r=-0.03 discovered
only after that whole run had been wasted).

Dataset: ProteinGym DMS assay RASK_HUMAN_Weng_2022_binding-DARPin_K55
(src/l38/data_cache/binding/), 24,873 non-indel variants of a single 188-residue
KRAS backbone, each with a real experimentally-measured binding score against
the DARPin K55 binder. Chosen over the other six cached binding assays because
it is the only one that is both large enough and short enough: SPIKE_SARS2,
DLG4_RAT and SPG1_STRSG have ZERO sequences at or under the harness's
MAX_SEQ_LEN=400 (their backbones are longer), and of the three that survive
length filtering, RASK is the largest (24,873 vs 6,137 CCR5 / 3,344 Q53Z42) and
the one where the proxy validates most strongly.

Proxy: mutational-sensitivity-weighted wildtype preservation, normalized by
overall sequence identity. Positions where mutations empirically destroy
binding (learned from the TRAIN split's labels only) get high weight; a
sequence scores well if it preserves the wildtype residue preferentially at
those binding-critical positions, over and above its overall fidelity to the
backbone. Validated against real labels before any steering run:

  proxy                          full r  full rho   test r  test rho
  weighted_id (raw)              +0.805    +0.814   +0.806    +0.813
  normalized (w - unweighted)    +0.795    +0.793   +0.797    +0.794
  blosum62_to_wt                 +0.273    +0.274   +0.257    +0.258
  net_charge                     +0.255    +0.266   +0.256    +0.267
  unweighted identity            +0.224    +0.235   +0.214    +0.224
  gravy                          +0.106    +0.079   +0.119    +0.091
  aromatic fraction              +0.014    +0.004   +0.010    +0.002

The NORMALIZED variant is used as the proxy rather than the marginally-stronger
raw one, because raw weighted identity is confoundable by generic
reconstruction fidelity: in this harness both the real-direction and
random-direction arms mask-fill 30% of positions, so a steering vector that
merely made generation more faithful to its input would raise raw weighted
identity without conferring any binding-specific capability. Subtracting
unweighted identity removes exactly that channel. Confirmed on simulated
random-position mutants carrying no binding signal, at increasing mutational
load (the regime generated sequences actually live in, far beyond this assay's
1-2 mutations):

  frac mutated   weighted_id   unweighted_id   normalized
          0.00        1.0000          1.0000      -0.0000
          0.10        0.9087          0.9093      -0.0006
          0.20        0.8157          0.8129      +0.0028
          0.30        0.7111          0.7169      -0.0058

Raw weighted identity tracks mutational load steeply (1.00 -> 0.71) while the
normalized variant stays pinned at ~0 with no trend -- it is blind to fidelity
alone and responds only to WHERE the preserved positions are.

Three further checks that this is real position-specific binding knowledge and
not an artifact, all run before the steering script was written:

  * Weight-shuffle control: permuting the learned weights across positions
    destroys the signal completely (r=-0.001, sd 0.101 over 10 shuffles) versus
    r=+0.805 with real weights. The predictive content is in WHICH positions
    are sensitive, not in similarity-to-wildtype generally.
  * Held-out POSITION generalization: fitting weights with 46 of 187 mutated
    positions entirely withheld still gives r=+0.541 / rho=+0.497 on variants
    at those unseen positions -- a real per-position scale, not a lookup table.
  * Mutational-load extrapolation: weights fit on single mutants only (n=3,084)
    transfer to double mutants (n=21,789) at r=+0.695 / rho=+0.733.

Rejected candidate, recorded rather than quietly dropped: a per-(position,
residue) mean-label-offset table ("PSSM") scored r=+0.854 on the full set but is
pure label memorization -- on the single-mutant assays it returns a CONSTANT on
held-out data, because no test mutant's exact (position, residue) pair ever
appears in the train table. High full-set r, zero generalization.
"""
from typing import Dict, Iterable, List, Sequence

import numpy as np


def parse_mutant_positions(mutant: str) -> List[int]:
    """Zero-based positions named by a ProteinGym `mutant` string, e.g.
    "A11C:D38C" -> [10, 37]. Tokens that don't parse as <wt><pos><mut> are
    skipped rather than raising, since ProteinGym's column occasionally carries
    non-substitution annotations.
    """
    positions = []
    for token in str(mutant).split(":"):
        if len(token) < 3:
            continue
        try:
            positions.append(int(token[1:-1]) - 1)
        except ValueError:
            continue
    return positions


def mutational_sensitivity_weights(
    mutants: Iterable[str], scores: Sequence[float], reference_length: int
) -> np.ndarray:
    """Per-position binding sensitivity, learned from real DMS labels.

    A position's raw sensitivity is how far mutations at it push the binding
    score BELOW the median variant; positions where mutation is neutral or
    helpful get zero. The result is L1-normalized so weights sum to 1, making
    the downstream proxy comparable across datasets and weight fits.

    MUST be fit on vector-building/train sequences only -- fitting on the eval
    split would leak eval labels into the scorer.
    """
    if reference_length <= 0:
        raise ValueError("reference_length must be positive")
    scores = np.asarray(list(scores), dtype=float)
    mutants = list(mutants)
    if len(mutants) != len(scores):
        raise ValueError("mutants and scores must have the same length")
    if len(scores) == 0:
        raise ValueError("mutational_sensitivity_weights requires at least one labeled variant")

    median = float(np.median(scores))
    per_position: Dict[int, List[float]] = {}
    for mutant, score in zip(mutants, scores):
        for position in parse_mutant_positions(mutant):
            if 0 <= position < reference_length:
                per_position.setdefault(position, []).append(float(score) - median)

    weights = np.zeros(reference_length, dtype=float)
    for position, effects in per_position.items():
        weights[position] = max(0.0, -float(np.mean(effects)))

    total = weights.sum()
    if total <= 0.0:
        raise ValueError(
            "no position showed a binding-reducing effect; weights would be all-zero "
            "(check that scores are oriented higher = better binding)"
        )
    return weights / total


def _aligned_length(sequence: str, reference: str) -> int:
    if len(sequence) == 0 or len(reference) == 0:
        raise ValueError("binding proxy requires non-empty sequence and reference")
    return min(len(sequence), len(reference))


def weighted_wildtype_preservation(sequence: str, reference: str, weights: np.ndarray) -> float:
    """Total sensitivity weight sitting on positions where `sequence` still
    carries the wildtype residue. 1.0 = exact wildtype, 0.0 = every sensitive
    position mutated.
    """
    n = _aligned_length(sequence, reference)
    if len(weights) < n:
        raise ValueError("weights must cover at least the aligned region")
    return float(sum(weights[i] for i in range(n) if sequence[i] == reference[i]))


def unweighted_identity(sequence: str, reference: str) -> float:
    """Plain fraction of aligned positions matching the reference -- the generic
    reconstruction-fidelity channel the proxy below subtracts out."""
    n = _aligned_length(sequence, reference)
    return sum(1 for i in range(n) if sequence[i] == reference[i]) / n


def binding_affinity_proxy(sequence: str, reference: str, weights: np.ndarray) -> float:
    """Higher = predicted better binding, matching this project's "higher score
    = more of the target property" convention (cf. L42's ivywrel_fraction,
    L51's aggregation_resistance_proxy, L55's disorder_proxy).

    Sensitivity-weighted wildtype preservation MINUS overall identity, so the
    score reflects preferential preservation at binding-critical positions
    rather than generic fidelity to the backbone. Validated at r=+0.795 (full)
    / +0.797 (held-out test) against real DMS binding labels -- see module
    docstring, including the weight-shuffle and mutational-load controls.
    """
    return weighted_wildtype_preservation(sequence, reference, weights) - unweighted_identity(
        sequence, reference
    )


def binding_affinity_proxy_excluding(
    sequence: str, reference: str, weights: np.ndarray, excluded_residues: frozenset
) -> float:
    """Residue-exclusion robustness check (L50 criterion 3), adapted for a
    position-aligned proxy.

    L42/L51/L55 implement this check by DELETING the dominant substituted
    residues from the string and rescoring. That is not available here: this
    proxy is aligned position-by-position against a reference backbone, so
    deleting characters would shift every downstream position and silently
    compare the wrong residues. The equivalent operation that preserves
    alignment is to MASK OUT the affected positions -- drop from both the
    weighted and unweighted terms any position where `sequence` carries an
    excluded residue -- which answers the same question the deletion form does:
    is the effect still there once the collapse residues stop contributing?
    """
    n = _aligned_length(sequence, reference)
    if len(weights) < n:
        raise ValueError("weights must cover at least the aligned region")
    kept = [i for i in range(n) if sequence[i] not in excluded_residues]
    if len(kept) == 0:
        raise ValueError("binding_affinity_proxy_excluding: no positions remain after exclusion")
    weighted = sum(weights[i] for i in kept if sequence[i] == reference[i])
    identity = sum(1 for i in kept if sequence[i] == reference[i]) / len(kept)
    return float(weighted) - identity
