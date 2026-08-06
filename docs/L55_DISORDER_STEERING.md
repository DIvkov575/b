# L55 — Intrinsic Disorder Steering: real, directionally robust effect; residue-robustness is seed-sensitive (2 of 3)

**Pre-registered against `docs/L50_CAPABILITY_GAIN_PROTOCOL.md`'s 6
criteria.** One of 5 parallel new-target attempts (L53 binding, L54
catalytic, L55 disorder, L56 immunogenicity, L57 expression yield).
**Do not report this as a clean unanimous PASS** — the effect direction
is robust across 3 independent seeds, but the residue-exclusion robustness
criterion specifically flips to KILL on 1 of 3. That seed-sensitivity
finding is itself one of this project's more useful methodological results
(see the seed table below), not a footnote to bury.

## Data and proxy

DisProt current release (`https://disprot.org/api/search`), 3324 entries
after canonical-AA cleaning, 1615 after length<=400 filtering. Per-sequence
scalar = fraction of residues inside a curated `type=="D"` consensus
region (validated to correlate r=0.84 with DisProt's own `disorder_content`
field). Proxy: mean TOP-IDP score (Campen et al. 2008) — a published,
literature-derived amino-acid disorder-propensity scale, not fit to this
project's own data.

Strongest-validated proxy of any target attempted before this session's
5-target sweep:

| check | result |
|---|---|
| full-set Pearson r | +0.449 (p=7e-81) |
| held-out test r | +0.482 (n=323) |
| spearman | +0.362 |
| partial r controlling for length | +0.428 (not a length confound) |
| holds within every length tertile | +0.342 / +0.466 / +0.515 |
| per-residue AUC (sliding window vs. real per-residue D/O labels, 354,556 residues) | 0.713 (window 21) |

## Steering results (SEED=0)

All 33 layers hooked. Same harness/controls as L42/L51/L54: matched-norm
random control, paired bootstrap, degeneracy filter, n=150 eval sequences
(ordered/low-disorder, held out from vector construction).

| alpha | real mean | random mean | diff | significant | n | degenerate (real) |
|---|---|---|---|---|---|---|
| 0.1 | 0.1266 | 0.1203 | +0.0061 | **yes** | 143 | 6/150 |
| 0.25 | 0.1392 | 0.1200 | +0.0190 | **yes** | 141 | 8/150 |
| 0.5 | 0.1691 | 0.1188 | +0.0497 | **yes** | 116 | 34/150 |
| 1.0 | 0.2372 | 0.1083 | +0.1271 | yes | 90 | 54/150 |
| 2.0 | 0.1304 | 0.1069 | -- | no (0 non-degenerate pairs) | 0 | 150/150 |
(baseline mean 0.1209, n=150, 7/150 degenerate)

Clean monotonic dose-response through the safe range (+0.006 -> +0.019 ->
+0.050). Note disorder-composition steering hits degeneracy earlier than
L54's catalytic target (34/150 already degenerate at alpha=0.5, vs. L54's
0/150 at the same alpha) — expected, since intrinsically disordered
sequences are genuinely low-complexity, and `is_degenerate_sequence`
(>25% single-AA) is measurably (if mildly, and conservatively in this
target's favor) biased toward flagging real disorder: on unmodified real
DisProt sequences it flags 1.5% of ordered ones but 6.7% of fully-disordered
ones.

**Residue-exclusion robustness (SEED=0):** dominant substituted residues
at alpha=0.5 are E and S. Excluding both: effect drops from +0.0497 to
+0.0178 (CI [0.011, 0.024], still significant) — 36% of magnitude retained.

## Seed-robustness check: direction replicates 3/3; residue-robustness replicates 2/3

Reran twice more with `SEED=1` and `SEED=2` (train/split and random-control
draw only — everything else identical):

| seed | diff @ 0.1/0.25/0.5 | criteria 1-2 (beats control, dose-response) | criterion 3 (residue-exclusion) | decision |
|---|---|---|---|---|
| 0 | +0.0061 / +0.0190 / +0.0497 | pass | **pass** — 36% retained (+0.0178, CI [0.011,0.024]) | **PASS** |
| 1 | +0.0037 / +0.0152 / +0.0373 | pass | **fail** — 10% retained (+0.0039, CI [-0.003,0.011], crosses zero) | **KILL** |
| 2 | +0.0053 / +0.0174 / +0.0435 | pass | **pass** — 42% retained (+0.0181, CI [0.012,0.025]) | **PASS** |

**The effect's existence and direction are seed-stable** (criteria 1-2 pass
on all 3 seeds, similar order of magnitude every time: +0.037-0.050 at
alpha=0.5). **The residue-exclusion robustness check specifically is not**
— it clears the bar on 2 of 3 seeds with 36-42% of the effect surviving,
but on seed 1 the surviving fraction drops to 10% and the CI crosses zero.

This is informative, not just noise: it means a meaningful fraction of the
disorder-steering effect's magnitude rides on E/S composition, and exactly
how much of that fraction survives exclusion depends on which specific
low/high-disorder sequences land in that run's vector-building split.
**Honest characterization: a real, directionally-robust effect that is
partially — not overwhelmingly — separable from E/S compositional
collapse.** Report as 2-of-3 seeds clearing the strict robustness bar, not
a unanimous PASS.

## Verdict: majority PASS (2/3 seeds), with an explicit seed-sensitivity caveat on criterion 3

| criterion | result |
|---|---|
| 1. beats random control, real CI | **PASS**, 3/3 seeds |
| 2. dose-response | **PASS**, monotonic across safe alphas, 3/3 seeds |
| 3. residue-exclusion robust | **PASS 2/3, KILL 1/3** — seed-sensitive |
| 4. proxy pre-validated | **PASS** (r=0.449, strongest of any target attempted) |
| 5. beats prior technique | N/A — no existing technique for this property |
| 6. adequately powered | **PASS**, n=150 |

## Vector-geometry relationship to L57 (expression yield)

Cosine similarity between this target's steering vector and L57's
(expression-yield) steering vector, computed layer-by-layer and on the
full 33-layer concatenation: **+0.30 overall, rising to +0.40-0.50 at
layers 30-32.** Not independent directions — L57's AMBIGUOUS result
(`docs/L57_EXPRESSION_STEERING.md`) is partially explained as a geometric
echo of this target's real effect, seen through a noisier, more
composition-collapse-prone proxy, rather than as independent new evidence
of a 6th steerable property. See that doc for the full analysis.

## What this is NOT

Not evidence this generalizes past the 25% single-AA degeneracy threshold
used throughout this project's harness — a near-miss KILL on a future rerun
should be re-checked with a looser threshold before being recorded, given
the threshold's documented (if modest) bias against this specific target.
Not a claim about which SPECIFIC residues beyond E/S drive the effect, or
whether a different pair would show the same seed-sensitivity.

## Cost

Three script runs (`plm_steering/l55_run_repro.py`, one per seed), each
5 alphas x 2 directions x 150 eval sequences, Apple Silicon MPS, ~4
minutes per run. Raw scores/sequences saved to
`plm_steering/l55_repro_out/`, `l55_repro_out_seed1/`, `l55_repro_out_seed2/`.
