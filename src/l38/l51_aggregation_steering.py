"""L51: extend the L42 activation-steering harness (validated on
thermostability, docs/L42_STEERING_REPRO.md) to a NEW target property --
aggregation propensity -- using a proxy that was checked against real
experimental labels BEFORE this run, unlike L43's GRAVY (which was found,
after the fact, to correlate only r=-0.03 with real solubility labels).

Dataset: cmartell/50C_Aggregation (HuggingFace), 13,853 real sequences with
experimental log2(fold-change in soluble fraction after 50C heat stress) --
more negative label = more aggregation-prone. Cleaned to plain amino-acid
sequences in src/l38/data_cache/aggregation/agg50_clean.csv.

Proxy: net charge (fraction of D/E minus fraction of K/R/H), validated
directly against this dataset's real labels: r=-0.20 (p~3e-129) on the full
set, r=-0.21 (p~1e-29) on the held-out test split alone -- confirmed NOT a
length confound. Decomposing further (negative charge r=+0.17, positive
charge r=-0.13) matches Lawrence et al. 2007 (JACS, "Supercharging Proteins
Can Impart Unusual Resilience") -- engineered net-negative charge increases
aggregation resistance, a real, independently-published mechanism, not a
formula invented for this project (the GRAVY lesson from L43).
"""
from typing import List

POSITIVE_RESIDUES = frozenset("KRH")
NEGATIVE_RESIDUES = frozenset("DE")


def net_charge(sequence: str) -> float:
    """(fraction of D/E) - (fraction of K/R/H) -- signed net charge proxy at
    physiological pH, computed compositionally (no model, no likelihood).
    """
    if len(sequence) == 0:
        raise ValueError("net_charge requires a non-empty sequence")
    pos = sum(1 for c in sequence if c in POSITIVE_RESIDUES)
    neg = sum(1 for c in sequence if c in NEGATIVE_RESIDUES)
    return (neg - pos) / len(sequence)


def aggregation_resistance_proxy(sequence: str) -> float:
    """Higher = more aggregation-RESISTANT (matches this project's "higher
    score = better" convention, e.g. L42's ivywrel_fraction). Checked
    directly against real labels (higher label = more resistant, per the
    dataset's log2 soluble-fraction convention): net_charge AS DEFINED ABOVE
    (neg - pos) already correlates POSITIVELY with real resistance
    (r=+0.20 on the full set, r=+0.21 on the held-out test split alone --
    see docs/L51_AGGREGATION_STEERING.md) -- i.e. more D/E-rich (more
    negatively charged) sequences are more resistant, matching Lawrence et
    al. 2007's supercharging result. No sign flip needed; net_charge IS the
    resistance proxy.
    """
    return net_charge(sequence)


def aggregation_resistance_proxy_excluding(sequence: str, excluded_residues: frozenset) -> float:
    """Same as aggregation_resistance_proxy but computed over a residue
    subset with excluded_residues removed first -- mirrors L42/L43's
    residue-exclusion robustness check, to rule out "the effect is just one
    residue's collapse rebranded" if steering causes a compositional
    collapse into a single D/E/K/R/H residue.
    """
    filtered = "".join(c for c in sequence if c not in excluded_residues)
    if len(filtered) == 0:
        raise ValueError("aggregation_resistance_proxy_excluding: no residues remain after exclusion")
    return aggregation_resistance_proxy(filtered)
