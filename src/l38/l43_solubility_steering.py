"""L43: extend the L42 activation-steering harness (validated on
thermostability, docs/L42_STEERING_REPRO.md) to a second target property --
solubility -- to test whether the reproduction generalizes or was specific
to thermostability. Reuses L42's difference-of-means vector construction,
degeneracy filter, and paired-bootstrap significance test unchanged; only
the target dataset and the model-free scoring function are new here.

Model-forward-pass pieces live in l43_run_repro.py (requires ESM2-650M on a
GPU). This module holds what's testable without a model: the GRAVY
solubility-proxy scorer.
"""
from typing import List

# Kyte & Doolittle (1982) hydropathy scale -- a purely compositional (no
# model, no likelihood) hydrophobicity index per residue. GRAVY (Grand
# AVerage of hYdropathy) is the standard summary: the mean hydropathy value
# across a sequence. NEGATIVE GRAVY (hydrophilic-leaning) is associated with
# higher aqueous solubility; POSITIVE GRAVY (hydrophobic-leaning) is
# associated with lower solubility / aggregation risk. Chosen for the same
# reason as L42's instability_index and ivywrel_fraction: it cannot be
# confounded by the generation model's own fluency judgments, since it never
# touches the model.
KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def gravy_score(sequence: str, scale: dict = KYTE_DOOLITTLE) -> float:
    """GRAVY (Grand Average of hYdropathy): mean Kyte-Doolittle hydropathy
    across a sequence. Lower (more negative) = more hydrophilic = proxies
    higher solubility. Unknown residues (e.g. 'X' from degenerate mask-fill
    output) contribute 0.0 -- neutral, matching the convention used for
    unknown dipeptides in L42's instability_index.
    """
    if len(sequence) == 0:
        raise ValueError("gravy_score requires a non-empty sequence")
    return sum(scale.get(c, 0.0) for c in sequence) / len(sequence)


def solubility_proxy(sequence: str) -> float:
    """Higher = more soluble-like, consistent sign convention with L42's
    ivywrel_fraction ("higher score = better" on the target property).
    Solubility is associated with LOW (negative) GRAVY, so negate it.
    """
    return -gravy_score(sequence)


def solubility_proxy_excluding(sequence: str, excluded_residues: frozenset) -> float:
    """Same as solubility_proxy but computed over a residue subset with
    excluded_residues removed first -- mirrors L42's leucine-exclusion check
    (docs/L42_STEERING_REPRO.md RESULTS v2), used to rule out "the effect is
    just one residue's collapse rebranded" for whichever residue the
    solubility steering vector's failure mode turns out to be, if any.
    """
    filtered = "".join(c for c in sequence if c not in excluded_residues)
    if len(filtered) == 0:
        raise ValueError("solubility_proxy_excluding: no residues remain after exclusion")
    return solubility_proxy(filtered)
