"""L57: extend the L42 activation-steering harness to a NEW target property --
soluble expression yield in E. coli -- with the scoring proxy validated
against real experimental labels BEFORE the run (L50 criterion 4), which is
exactly what L43 failed to do with GRAVY (found only r=-0.03 after the fact).

Dataset: eSol (Niwa et al. 2009, PNAS 106:4201 -- chaperone-free PURE
cell-free translation of the E. coli ORFeome), via HuggingFace AI4Protein/eSOL,
cached to src/l38/data_cache/expression/esol_clean.csv. 3101 distinct E. coli
proteins, CONTINUOUS label in [0,1] = soluble fraction of expressed protein;
2337 remain at <=400 residues.

Distinct from L43's cached hazemessam/solubility, verified empirically rather
than assumed: 441 sequences appear in both, and on those 441 the two labels
are UNCORRELATED (point-biserial r=0.048, p=0.32). Same word "solubility,"
statistically orthogonal measurement -- L43's is a binary
soluble/inclusion-body call on heterologous constructs (2460 of its sequences
carry an MAHHHHHH purification tag; eSol has zero), eSol's is a continuous
per-protein soluble fraction from a defined in-vitro translation system.
Testing this proxy/dataset is therefore new evidence, not a rerun of L43.

Proxy: absolute charge average, |(K+R) - (D+E)| / length.

Validated against eSol's real labels on the length-filtered set (n=2337):
Pearson r=+0.305 (p=2.4e-51), Spearman +0.284; on the 228-sequence held-out
test split alone, Pearson r=+0.337 (p=1.8e-07), Spearman +0.260. An order of
magnitude above L43's disqualifying r=-0.03, and above L51's accepted r=0.20.

Not a formula invented for this project: absolute charge average is one of the
two terms in Wilkinson & Harrison's (1991, Bio/Technology 9:443) published
recombinant-solubility discriminant, where HIGH absolute charge favors
solubility. Both of that discriminant's terms point the direction the
published formula says they should in this dataset (|charge| r=+0.30 toward
soluble; turn-forming N/G/P/S fraction r=-0.10, i.e. toward insoluble), and
the full composite validates too (r=+0.288). The single |charge| term is used
here because it is the stronger and simpler half.

Deliberately NOT hydrophobicity in disguise: r=-0.17 against GRAVY, so this
is not L43's dead proxy relabeled. Also near-orthogonal to L51's signed
net_charge (r=-0.001) -- signed charge asks "which way," this asks "how far
from neutral," and only the latter tracks eSol's label.

Length confound, checked not waved off: raw sequence length is itself the
single strongest correlate of eSol solubility (r=+0.34, shorter = more
soluble). This proxy survives partialling length out (r=+0.228), and more
decisively, length CANNOT confound the steering experiment at all: masked-fill
generation substitutes residues in place and never changes sequence length,
so every arm is compared at identical length.
"""
POSITIVE_RESIDUES = frozenset("KR")
NEGATIVE_RESIDUES = frozenset("DE")


def absolute_charge_average(sequence: str) -> float:
    """|(#K + #R) - (#D + #E)| / length -- distance of net charge from
    neutrality, per residue. Purely compositional: no model, no likelihood,
    so it cannot be confounded by ESM2's own fluency judgments about the
    sequences it generated.
    """
    if len(sequence) == 0:
        raise ValueError("absolute_charge_average requires a non-empty sequence")
    pos = sum(1 for c in sequence if c in POSITIVE_RESIDUES)
    neg = sum(1 for c in sequence if c in NEGATIVE_RESIDUES)
    return abs(pos - neg) / len(sequence)


def expression_yield_proxy(sequence: str) -> float:
    """Higher = higher predicted soluble expression yield, matching this
    project's "higher score = better" convention (cf. L42's ivywrel_fraction).
    No sign flip needed: absolute_charge_average already correlates POSITIVELY
    with eSol's real soluble-fraction label (r=+0.305 full set, +0.337
    held-out test split), the direction Wilkinson & Harrison 1991 predicts.
    """
    return absolute_charge_average(sequence)


def expression_yield_proxy_excluding(sequence: str, excluded_residues: frozenset) -> float:
    """Same proxy over a residue subset with excluded_residues removed first --
    L42/L43's residue-exclusion robustness check (L50 criterion 3), to rule
    out "the effect is one residue's compositional collapse rebranded," the
    artifact that produced L42 v1's poly-leucine false PASS and L43's A/G
    false significance.
    """
    filtered = "".join(c for c in sequence if c not in excluded_residues)
    if len(filtered) == 0:
        raise ValueError("expression_yield_proxy_excluding: no residues remain after exclusion")
    return expression_yield_proxy(filtered)
