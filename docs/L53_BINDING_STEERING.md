# L53 — Binding Affinity Steering: KILL, despite the best-validated proxy in the project

**Pre-registered against `docs/L50_CAPABILITY_GAIN_PROTOCOL.md`'s 6 criteria.**
One of 5 parallel new-target attempts (L53 binding, L54 catalytic, L55
disorder, L56 immunogenicity, L57 expression yield). This is the sharpest
single data point in the whole project for "proxy validity does not predict
steerability": this target's proxy is the strongest-validated of any target
tried (r=0.795-0.806 held-out — nearly 4x L54's r=0.22), and the steering
effect is a flat, unambiguous null.

## Data

ProteinGym DMS assay `RASK_HUMAN_Weng_2022_binding-DARPin_K55`
(`plm_steering/data_cache/binding/`), 24,873 non-indel variants of a single
188-residue KRAS backbone, each with a real experimentally-measured binding
score against the DARPin K55 binder. Chosen over 6 other cached binding
assays because it is the only one both large enough and short enough after
length-filtering to <=400 residues (see
`plm_steering/l53_binding_affinity_steering.py`'s docstring for the full
comparison table). Split 17,411 vector-building pool / 7,462 eval pool.

## Proxy: mutational-sensitivity-weighted wildtype preservation

Per-position binding-sensitivity weights learned from the TRAIN split's
real labels only (a position's weight = how far mutating it pushes the
score below the median, zero for neutral/helpful positions), then a
generated sequence scores well if it preferentially preserves the wildtype
residue at binding-CRITICAL positions specifically (not just generic
fidelity to the backbone — the raw weighted-identity score is corrected by
subtracting plain unweighted identity, since both arms of this harness
mask-fill 30% of positions and a direction that just made generation more
faithful in general would inflate raw weighted identity without any real
binding-specific signal).

Validated far more rigorously than any other target in this project:

| check | result |
|---|---|
| full-set r | +0.805 (rho +0.814) |
| held-out test r | +0.797 (rho +0.794) |
| weight-shuffle null control | r=-0.001 (sd 0.101 over 10 shuffles) — confirms the signal is WHICH positions are sensitive, not generic similarity |
| held-out POSITION generalization (46/187 positions withheld) | r=+0.541 on unseen positions — a real per-position scale, not a lookup table |
| mutational-load extrapolation (fit on single mutants, tested on doubles) | r=+0.695 |
| rejected: raw per-(position,residue) PSSM lookup | r=+0.854 full-set but returns a CONSTANT on held-out data — pure memorization, explicitly rejected rather than used despite the higher headline number |

## Steering results: flat null at every alpha

| alpha | real mean | random mean | diff | significant | n |
|---|---|---|---|---|---|
| 0.1 | 0.04854 | 0.04844 | +0.0001 | no | 150 |
| 0.25 | 0.04854 | 0.04852 | +0.00001 | no | 150 |
| 0.5 | 0.04818 | 0.04864 | -0.0005 | no | 150 |
| 1.0 | 0.04839 | 0.04869 | -0.0003 | no | 150 |
| 2.0 | 0.04864 | 0.04841 | +0.0002 | no | 150 |
(baseline mean 0.04829, n=150, 0 degenerate at every alpha — no collapse
confound anywhere in this run, unlike most other targets)

The real-direction score is essentially frozen across the entire alpha
range (0.0482-0.0486), statistically indistinguishable from the
matched-norm random control at every single point, with no dose-response
in either direction. This is not a borderline or alpha-sensitive result —
it is a clean, unambiguous null across all 5 tested alphas.

## Verdict: KILL

| criterion | result |
|---|---|
| 1. beats random control, real CI | **fails** at every alpha |
| 2. dose/scale-response | **fails** — flat, no trend |
| 3. residue-exclusion robust | not applicable — no significant effect to check robustness of |
| 4. proxy pre-validated | pass (r=0.80, by far the strongest in the project) |
| 6. adequately powered | pass (n=150, zero degeneracy issues) |

## Why: a mechanistic hypothesis, checked directly

Compositional distance between the vector-building low/high groups was
measured directly (mean per-residue amino-acid-composition L2 distance
between the 150 low-binding and 150 high-binding sequences used to build
the steering vector): **0.0033** for this target, versus **0.023-0.045**
for L54/L55/L57 (the cross-protein, non-single-backbone datasets). This
target's low/high groups are 1-2-residue point mutants of the SAME
188-residue backbone — by construction, there is very little compositional
separation for a difference-of-means vector to find, independent of
whether binding affinity is causally steerable in ESM2-650M at all. The
proxy captures real position-specific binding information (see the
validation table above), but a difference-of-means vector built from
near-identical sequences may simply have very little signal to work with,
regardless of how well the SCORING function correlates with the real
label.

This does not rule out binding affinity as steerable in principle — a
cross-protein binding dataset (analogous to L54's DLKcat) was not tried
here, and would be the natural next test before concluding the *property*
itself is unsteerable rather than this specific *single-backbone DMS
setup*.

## What this means for the project's central finding

This is the single sharpest data point for "proxy validity does not
predict causal steerability": r=0.80 (best in the project) -> null, versus
L54's r=0.22 (weakest of the four runnable targets) -> a real, 3-seed-
replicated PASS (`docs/L54_CATALYTIC_STEERING.md`). Proxy-validation
strength and steering effect size are not even weakly monotonically
related in this data.

## Cost

Single script run, `plm_steering/l53_run_repro.py`, 5 alphas x 2 directions
x 150 eval sequences, Apple Silicon MPS, ~4 minutes. Raw scores/sequences
saved to `plm_steering/l53_repro_out/results.json`.
