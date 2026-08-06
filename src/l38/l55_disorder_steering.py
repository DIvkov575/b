"""L55: extend the L42 activation-steering harness (validated on
thermostability, docs/L42_STEERING_REPRO.md) to a NEW target property --
intrinsic disorder (IDR) content -- using a proxy validated against real
per-residue disorder annotations BEFORE the run, per L50 criterion 4.

Dataset: DisProt current release (https://disprot.org/api/search), 3324
entries with canonical-AA-only sequences, cleaned to
src/l38/data_cache/disorder/disprot_clean.csv. Unlike every prior target in
this arc (thermostability/solubility/aggregation, all one scalar per whole
sequence), disorder has real GROUND TRUTH PER RESIDUE: each entry carries
DisProt's curated consensus disordered regions, reduced here to a
per-sequence scalar (fraction of residues inside a curated disordered
region) for the difference-of-means split, and retained per-residue for the
extra AUC validation below. The derived per-sequence fraction correlates
r=0.84 with DisProt's own `disorder_content` field (differences are
ambiguous/non-"D" region types), confirming the reduction is faithful.

Proxy: TOP-IDP scale (Campen et al. 2008, Protein Pept Lett 15:956-963,
"TOP-IDP-scale: a new amino acid scale measuring propensity for intrinsic
disorder") -- a published, experimentally-derived per-residue disorder
propensity scale, mean-pooled over the sequence. Chosen over three
alternatives after validating ALL of them against real labels first:

  proxy              pearson r   spearman   test-split r
  TOP-IDP              +0.449     +0.362      +0.482
  simple_comp          +0.443     +0.359      +0.446
  FoldIndex (flipped)  +0.385     +0.334      +0.372
  neg. hydropathy      +0.372     +0.311      +0.382

TOP-IDP wins and is NOT a length confound: partial correlation controlling
for sequence length is r=+0.428 (vs raw +0.449), and it holds within every
length tertile (+0.342 / +0.466 / +0.515). Per-residue rigor check (richer
than any prior target in this arc allowed): sliding-window mean TOP-IDP vs.
DisProt's per-residue D/O labels over 354,556 residues gives AUC=0.713
(window=21; 0.710 at 31, 0.700 at 41) -- the proxy discriminates
disordered from ordered residues, not just whole-sequence averages.

This is ~2.2x the effect size of L51's pre-validated aggregation proxy
(r=+0.20) and a different universe from L43's GRAVY (r=-0.03, discovered
only after that run had already been wasted).
"""

# Campen et al. 2008, TOP-IDP scale. Higher = more disorder-promoting.
# Ordering here reflects the published scale: P/E/K most disorder-promoting,
# W/F/Y/I most order-promoting.
TOP_IDP_SCALE = {
    "W": -0.884, "F": -0.697, "Y": -0.510, "I": -0.486, "M": -0.397,
    "L": -0.326, "V": -0.121, "N": 0.007, "C": 0.020, "T": 0.059,
    "A": 0.060, "G": 0.166, "R": 0.180, "D": 0.192, "H": 0.303,
    "Q": 0.318, "S": 0.341, "K": 0.586, "E": 0.736, "P": 0.987,
}


def top_idp_score(sequence: str) -> float:
    """Mean TOP-IDP propensity over a sequence. Higher = more predicted
    intrinsic disorder. Compositional only -- no model, no likelihood, so it
    cannot be circular with the ESM2 model being steered.

    Non-canonical characters (X, B, Z, U, O, and any tokenizer artifact) are
    skipped rather than assigned a propensity, since the published scale
    defines values only for the 20 canonical amino acids.
    """
    values = [TOP_IDP_SCALE[c] for c in sequence if c in TOP_IDP_SCALE]
    if len(values) == 0:
        raise ValueError("top_idp_score requires at least one canonical amino acid")
    return sum(values) / len(values)


def disorder_proxy(sequence: str) -> float:
    """Higher = more intrinsically disordered, matching this project's
    "higher score = better/more of the target property" convention (cf.
    L42's ivywrel_fraction, L51's aggregation_resistance_proxy).

    TOP-IDP already points this way against real labels (r=+0.449, see
    module docstring), so no sign flip is needed.
    """
    return top_idp_score(sequence)


def disorder_proxy_excluding(sequence: str, excluded_residues: frozenset) -> float:
    """Same as disorder_proxy but computed after removing excluded_residues --
    mirrors L42/L43/L51's residue-exclusion robustness check, to rule out
    "the effect is just a collapse into one disorder-promoting residue
    (P/E/K) rebranded as a real capability."
    """
    filtered = "".join(c for c in sequence if c not in excluded_residues)
    if len(filtered) == 0:
        raise ValueError("disorder_proxy_excluding: no residues remain after exclusion")
    return disorder_proxy(filtered)
