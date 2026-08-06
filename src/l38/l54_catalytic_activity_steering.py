"""L54: extend the L42 activation-steering harness to a NEW target property --
enzyme catalytic activity (turnover number, kcat) -- with the scoring proxy
validated against real experimental labels BEFORE the steering run, per
docs/L50_CAPABILITY_GAIN_PROTOCOL.md criterion 4.

Dataset: DLKcat (src/l38/data_cache/catalytic/dlkcat_wt_mut.json), 17,010
enzyme/substrate/kcat records from BRENDA+SABIO-RK. Filtered to sequences
<=400 residues with canonical residues and kcat>0, deduplicated by sequence
(median log10 kcat where a sequence appears with several substrates): 4,370
unique enzymes spanning EC classes 1-7 and many organisms.

Chosen over the two available single-backbone catalytic DMS assays
(AMIE_PSEAE_Wrenbeck_2017, OXDA_RHOTO_Vanella_2023) deliberately. Those are
6k+ single point mutants of ONE 346/364-residue enzyme, so every compositional
proxy varies over ~1/346 of the sequence: BLOSUM62-similarity-to-wildtype
reaches r=+0.23 on AMIE, but its entire validated range spans a
one-substitution window, whereas 30%-masked generation rewrites ~100
positions. Validating a proxy on 1-mutation variation and then scoring
100-mutation generations would be exactly L43's GRAVY mistake in a new
costume. DLKcat's cross-protein spread is the regime the steering eval
actually operates in.

Proxy: glycine fraction minus arginine fraction, validated against this
dataset's real log10(kcat):
  full set        r=+0.220 (p=8e-49, n=4370), spearman +0.22
  held-out test   r=+0.212 (p=1e-14, n=1311), spearman +0.21
  length-residualized  r=+0.223  (not a length confound; the label itself is
                                  uncorrelated with length, r=-0.02)
  wildtype-only subset r=+0.146 (p=1e-09, n=1738) -- signal is not an
                                  artifact of engineered mutants
  within-EC-class      r=+0.12 (EC1) to +0.35 (EC4), significant in 4 of the
                                  5 classes with n>=100 -- not a fold or
                                  EC-class confound
Mechanism is the published activity-stability tradeoff: high-turnover
(notably cold-adapted/psychrophilic) enzymes buy active-site conformational
flexibility with glycine-rich, arginine- and salt-bridge-poor sequences,
while rigid thermostable enzymes trade turnover for Arg-mediated salt bridges
and ion pairs (Fields 2001, Comp Biochem Physiol; Siddiqui et al. 2006, Extremophiles;
Berezovsky & Shakhnovich 2005, PNAS on Arg/ion-pair enrichment in thermophiles).
Consistent with that literature, L42's IVYWREL thermostability proxy
correlates NEGATIVELY with kcat here (r=-0.12) -- the tradeoff appears in the
data with the sign the mechanism predicts, so this is a documented
biophysical correlate rather than a formula fit to this project's numbers.

Weaker alternatives measured on the same held-out split and rejected in
favor of gly-minus-arg: glycine fraction alone (test r=+0.193), broad
catalytic-residue fraction H/C/D/E/S/K/R/Y/N/T/W (r=-0.149), aromatic
fraction (-0.133), charged fraction D/E/K/R (-0.109), GRAVY (+0.092),
Arg/(Arg+Lys) ratio (+0.075), net charge (-0.009, null), sequence length
(-0.022, null). Larger hand-built composites (adding Ala, IVYWREL, or
normalizing Arg by Arg+Lys) all scored below the two-term form.
"""

GLYCINE = frozenset("G")
ARGININE = frozenset("R")


def _fraction(sequence: str, residues: frozenset) -> float:
    return sum(1 for c in sequence if c in residues) / len(sequence)


def gly_minus_arg(sequence: str) -> float:
    """(fraction of G) - (fraction of R) -- purely compositional flexibility
    proxy, no model and no likelihood involved (same discipline as L42's
    IVYWREL and L51's net charge).
    """
    if len(sequence) == 0:
        raise ValueError("gly_minus_arg requires a non-empty sequence")
    return _fraction(sequence, GLYCINE) - _fraction(sequence, ARGININE)


def catalytic_activity_proxy(sequence: str) -> float:
    """Higher = predicted HIGHER catalytic turnover, matching this project's
    "higher score = better" convention (L42's ivywrel_fraction, L51's
    aggregation_resistance_proxy). gly_minus_arg already correlates POSITIVELY
    with real log10(kcat) (r=+0.220 full / +0.212 held-out test; see module
    docstring and docs/L54_CATALYTIC_STEERING.md), so no sign flip is needed.
    """
    return gly_minus_arg(sequence)


def catalytic_activity_proxy_excluding(sequence: str, excluded_residues: frozenset) -> float:
    """Same as catalytic_activity_proxy but computed after deleting
    excluded_residues -- L42/L43/L51's residue-exclusion robustness check,
    which rules out "the effect is one residue's compositional collapse
    rebranded."

    Note the proxy is a two-residue contrast, so excluding G or R removes one
    of its own terms; that is intentional and is the strictest possible form
    of this check (if the whole effect is steering pumping out glycine, the
    G-excluded score cannot reproduce it).
    """
    filtered = "".join(c for c in sequence if c not in excluded_residues)
    if len(filtered) == 0:
        raise ValueError("catalytic_activity_proxy_excluding: no residues remain after exclusion")
    return catalytic_activity_proxy(filtered)
