# L43 — Extend the L42 Steering Harness to a Second Target: Solubility

**Pre-registered protocol.** Locked 2026-07-26, before any run.

## STATUS (2026-07-27): run complete — AMBIGUOUS, does not confirm generalization

AWS credentials for the EC2 instance expired mid-session and were never
refreshed. Rather than wait, the run was done **locally on Apple Silicon
(M3 Pro) via PyTorch MPS** instead of CUDA — `l42_run_repro.py` and
`l43_run_repro.py` were patched to prefer `mps` when `cuda` isn't available
(one-line change, no other CUDA-specific code existed in either script).
Sanity check: rerunning L42 on MPS reproduced its EC2/A10G result exactly
(identical point estimates and CIs at every alpha), confirming MPS is a
valid substitute for this workload, not just "fast but wrong."

**Result: AMBIGUOUS, per this doc's own pre-registered rule.** Full run
output: `src/l38/l43_repro_out/results.json`.

| alpha | real vs random diff | 95% CI | significant |
|-------|---------------------|--------|--------------|
| 0.1   | -0.0013 | [-0.0107, 0.0080] | no |
| 0.25  | +0.0136 | [-0.0030, 0.0297] | no |
| 0.5   | +0.0098 | [-0.0152, 0.0332] | no |
| 1.0   | -0.0263 | [-0.0727, 0.0199] | no |
| 2.0   | -0.0960 | [-0.1629, -0.0279] | **yes** |

Unlike L42's clean monotonic dose-response (effect grew smoothly and
consistently with alpha: +0.007 → +0.022 → +0.050), L43's effect **flips
sign incoherently across alphas** with no trend, and the one alpha that
clears significance (2.0) is also the **wrong sign** — steering with the
"toward soluble" vector made sequences score as LESS soluble than the
random control, not more.

**The residue-exclusion robustness check (this protocol's own AMBIGUOUS
clause) confirms this is an artifact, not a real effect.** Substitution
analysis at alpha=2.0 showed the dominant compositional shift is toward
alanine and glycine (A: 776 substitutions-to, G: 711 — the two most common
targets by a wide margin). Rescoring with A/G excluded from the GRAVY
calculation **collapses the alpha=2.0 effect from -0.096 (significant) to
-0.0095 (not significant)** — the entire "effect" is explained by a
compositional shift toward two small, chemically mild residues under heavy
steering, not a genuine solubility-relevant shift. This is the same
category of artifact as L42's leucine collapse, just with different
residues and (this time) correctly caught before being reported as a win.

**Per the pre-registered rule: this is AMBIGUOUS, not PASS.** The automated
verdict in `results.json` says `"decision": "PASS"` because the script's
verdict logic (copied from L42) only checks "does any alpha clear
significance," and does not itself run the residue-exclusion check — that
check has to be applied manually per this doc's own AMBIGUOUS clause, and
was applied here. **Do not trust the automated `"decision"` field in
`l43_repro_out/results.json` at face value** — this is exactly the failure
mode the AMBIGUOUS clause exists to catch, and it worked.

**Conclusion: L42's reproduction does NOT straightforwardly generalize to
solubility with this exact recipe.** This does not mean solubility can't be
steered — it means the difference-of-means vector, generation setup, and
scoring proxy that worked for thermostability produce an artifact-dominated
result for solubility at the alpha range where the effect would need to be
large enough to detect. Two live possibilities, neither tested yet: (a) the
low end (0.25–0.5) has a small, inconsistent hint (63% of sequences move
positive at 0.25, mean diff +0.014, not significant) that might resolve
with a larger eval set rather than the current n=60; (b) GRAVY specifically
may be a poor proxy for how ESM2's solubility-associated activations
actually organize residue composition — unlike IVYWREL (which was verified
by checking real thermophile/mesophile literature before trusting it), the
GRAVY-based split here was not independently checked against what
alanine/glycine enrichment actually means for aqueous solubility in the
literature before this run.

## Why this exists

L42 (docs/L42_STEERING_REPRO.md) reproduced Huang et al.'s difference-of-
means activation steering on ESM2-650M for thermostability, after fixing a
real decoding-collapse artifact. That result validated the harness for
thermostability specifically, but doesn't establish whether the harness
generalizes to a different target property or whether the thermostability
result was itself a fluke tied to that particular dataset/split. A follow-
up literature survey (2026-07-24) also surfaced that continuous,
compositionally-grounded properties are the one repeatedly-successful
recipe for single-direction steering in the field, and solubility is the
most direct next test of that recipe using infrastructure that's already
built and validated.

## The claim under test

**H1.** A difference-of-means steering vector built from ESM2-650M
activations of soluble vs. insoluble sequences (real experimental labels,
hazemessam/solubility dataset) increases a downstream solubility proxy
score when added to the residual stream during masked-marginal generation,
relative to both an unsteered baseline and a matched-norm random-direction
control.

This is a generalization check, not a new scientific claim on its own:
success buys confidence that L42's reproduction wasn't thermostability-
specific; failure would suggest the harness (or the difference-of-means
technique at this scale) is narrower than L42 alone implied.

## Method (reuses L42's validated recipe unchanged except where noted)

- **Model:** `facebook/esm2_t33_650M_UR50D` — identical to L42.
- **Target property:** solubility. **Dataset:** `hazemessam/solubility`
  (HuggingFace, same author/format convention as L42's `hazemessam/
  meltome`), real experimental soluble/insoluble binary labels, ~62K
  sequences. Unlike L42's continuous Tm labels (requiring a percentile
  split), this dataset's ground truth is already binary — used directly as
  the high/low group split.
- **Steering vector construction:** difference-of-means, identical to L42 —
  mean per-layer activation (averaged across tokens) over the soluble group
  minus the insoluble group, across all 33 transformer layers.
- **Generation:** identical to L42 — single-shot masked-marginal fill,
  `mask_fraction=0.3`, `alpha` in `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`. Both
  values are REUSED, not re-derived, from L42's empirical tuning on this
  same model/generation setup — there's no a priori reason the collapse
  point would be property-specific rather than model/generation-specific,
  but this is an assumption, not a re-verified fact. If the results below
  show degeneracy behaving very differently from L42's pattern (e.g.
  collapse starting much earlier or not at all through alpha=2.0), that
  itself is worth flagging rather than silently trusted.
- **Scoring:** GRAVY (Grand Average of hYdropathy, Kyte & Doolittle 1982) —
  a purely compositional, model-free hydrophobicity index, negated so
  higher = more soluble-like. Chosen for the same reason as L42's
  `ivywrel_fraction`: cannot be confounded by the generation model's own
  fluency judgments, and is NOT derived from anything observed in this
  project's own generated sequences (it's a 1982 literature-standard scale).
  See `src/l38/l43_solubility_steering.py`.
- **Degeneracy filter:** identical to L42's `is_degenerate_sequence`
  (>25% single-residue frequency) — applied BEFORE scoring, same
  `MIN_NONDEGENERATE_PAIRS=30` trust guard on the bootstrap.
- **Significance test:** identical to L42's direct real-vs-random paired
  bootstrap (`paired_bootstrap_mean_diff`), not two separate vs.-baseline
  tests.
- **Leucine-exclusion-style robustness check:** if a significant effect is
  found, rerun the score with whichever residue(s) turn out to dominate the
  steering direction's shift excluded (`solubility_proxy_excluding`), to
  rule out "same collapse artifact, different name" — the exact check that
  saved L42 from a false positive.

## PASS / KILL / AMBIGUOUS rule (pre-registered, identical structure to L42)

- **PASS:** real-direction steering significantly beats the random-direction
  control (bootstrap CI excludes 0) at at least one alpha with
  `>= MIN_NONDEGENERATE_PAIRS` surviving non-degenerate sequence pairs, AND
  the effect survives a residue-exclusion robustness check if applicable.
- **KILL:** no alpha shows a significant real-vs-random effect with enough
  surviving non-degenerate pairs.
- **AMBIGUOUS:** a significant effect is found but does NOT survive the
  residue-exclusion check (i.e., it's an artifact of the same kind that
  produced L42's false positive) — treat as informative but not sufficient
  to claim generalization.

## What this is NOT

Not a new scientific claim about solubility biology on its own — a
successful PASS here means "the L42 harness generalizes to a second
property," which is valuable for deciding whether to trust this pipeline on
a genuinely novel target, not a publishable solubility-design result by
itself.

## Follow-up (2026-08-03): larger eval set + GRAVY proxy quality check

Two open leads from the original run: (a) the low-alpha hint (63% of
sequences moved positive at alpha=0.25, mean diff +0.014, not significant
at n=60) might resolve with more statistical power; (b) GRAVY was never
checked against real solubility labels before use, unlike IVYWREL (checked
against thermophile/mesophile literature in L42).

**(a) Larger eval set (n=60 → n≈287-288 non-degenerate pairs, 5x):**
`src/l38/l43_run_repro_v2_largeeval.py`, alphas restricted to [0.1, 0.25,
0.5] (the established non-collapsing range). Result: **still not
significant at any alpha.**

| alpha | real vs random diff | 95% CI | significant | % sequences real > random |
|-------|---------------------|--------|--------------|---------------------------|
| 0.1   | +0.0022 | [-0.0019, 0.0062] | no | 45.1% |
| 0.25  | +0.0059 | [-0.0012, 0.0130] | no | 53.0% |
| 0.5   | -0.0009 | [-0.0118, 0.0100] | no | 49.1% |

The low-alpha hint does NOT resolve with 5x the sample — if anything, the
point estimate at alpha=0.25 stayed the same order of magnitude
(+0.006 vs. the original +0.014) while the CI tightened enough to rule out
anything but a very small effect. No dose-response pattern (0.1 < 0.25 but
0.5 drops below both). At n≈288 this is a well-powered null, not an
underpowered one — the original "might resolve with more data" hypothesis
is now closed.

**(b) GRAVY proxy quality, checked directly against real labels (not
assumed):** computed GRAVY for all 62,478 sequences in the training set and
compared against the real soluble/insoluble labels.

| | soluble (n=26,075) | insoluble (n=36,403) |
|---|---|---|
| mean GRAVY | -0.342 | -0.325 |

Mann-Whitney U p=1.9e-11 (real, given the large n) but point-biserial
correlation **r=-0.030** — the direction is correct (soluble sequences
really do skew slightly more hydrophilic, matching Kyte-Doolittle's
intended use) but the effect size is negligible. GRAVY explains under 0.1%
of the variance in this dataset's solubility label. **This resolves lead
(b): GRAVY is a statistically real but practically very weak proxy for
solubility in this dataset** — a plausible reason a real underlying
steering effect (if one exists) would be undetectable through this
particular scoring function, independent of whether the steering itself
works.

## Revised conclusion

Both open leads are now closed, not just left open. The harness does not
generalize to solubility with GRAVY as the proxy, and this is no longer
attributable to insufficient sample size (a) or an untested-but-plausible
proxy concern (b) — it's a proxy-quality ceiling: GRAVY's real correlation
with solubility in this dataset (r=-0.03) is simply too weak to detect
through a steering-vs-random-control comparison at any sample size this
project would practically run. A genuine next attempt at solubility
steering would need either a better proxy (e.g. a frozen-embedding
classifier probe trained on a disjoint split, mirroring L39's/L41's Gate 3
methodology, rather than a fixed literature formula) or an experimentally
stronger dataset where the solubility signal is compositionally larger.
