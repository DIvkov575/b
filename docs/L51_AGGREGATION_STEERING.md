# L51 — Aggregation Resistance Steering: KILL (script's own "PASS" is wrong)

**Manually corrected verdict: KILL.** `l51_repro_out/results.json`'s own
`"decision": "PASS"` field is misleading — it only checks "did any alpha
clear a significant real-vs-random CI," the same shallow rule that produced
L42 v1's false PASS and L43's original false significance before those were
caught by additional checks. Applying the actual `docs/L50_CAPABILITY_GAIN_PROTOCOL.md`
criteria by hand overturns it.

## Data and proxy

`cmartell/50C_Aggregation` (HuggingFace), 13,853 real sequences with
experimental aggregation-propensity labels (50°C aggregation assay).
Proxy: net charge (validated at r=+0.20 against real labels — the
best-validated proxy in the project at the time, before L53-L57's later
sweep found stronger and weaker alternatives). A double-negation sign bug
in the proxy was found and fixed before this run.

## Results

| alpha | real mean | random mean | diff | significant | n | degenerate (real) |
|---|---|---|---|---|---|---|
| 0.1 | -0.0284 | -0.0283 | -0.0003 | no | 141 | 6/150 |
| 0.25 | -0.0293 | -0.0278 | -0.0015 | no | 139 | 8/150 |
| 0.5 | -0.0324 | -0.0303 | -0.0012 | no | 136 | 11/150 |
| **1.0** | -0.0197 | -0.0308 | **+0.0197** | **yes** | 91 | **50/150** |
| 2.0 | -0.0301 | -0.0262 | -0.0026 | no | 93 | 44/150 |
(baseline mean -0.0284, n=150, 8/150 degenerate)

**Only alpha=1.0 is significant, and it is not a safe alpha.** Per the
lesson later formalized in `docs/L52_LAYER_SUBSET_STEERING.md`'s "Critical
correction" (alpha>=1.0 is exactly the regime where this harness's
single-shot argmax mask-fill degenerates independent of any real steering
effect — see `docs/L42_STEERING_REPRO.md`'s established safe range,
alpha in [0.1, 0.5]): at alpha=1.0, **50 of 150 real-direction sequences are
already degenerate** (1/3 of the eval set), and the CI is computed only
over the 91 surviving pairs. There is no dose-response across the safe
range 0.1→0.25→0.5 — the effect is flat/negative-noise there, then a single
value flips positive exactly where degeneracy is worst. This is the same
"one lucky value outside the safe window" pattern the L52 bug produced,
not a coherent trend.

The residue-exclusion "robustness check" (excluding A, K at alpha=1.0,
diff grows to +0.0253, still significant) does not rescue this: it is
checked at the same untrustworthy alpha, so it inherits the same problem
rather than independently confirming anything.

## Verdict: KILL

| criterion | result |
|---|---|
| 1. beats random control, real CI | fails outside alpha=1.0; the one pass is at an unsafe alpha |
| 2. dose/scale-response | **fails** — no trend across 0.1/0.25/0.5, isolated spike at 1.0 |
| 3. residue-exclusion robust | not meaningfully checked (only run at the unsafe alpha) |
| 4. proxy pre-validated | pass (r=+0.20) |
| 6. adequately powered | pass (n=150) |

Same false-positive shape as L42 v1 (poly-leucine collapse fooling a naive
metric) and L43 (alpha=2.0 false significance) — a proxy that validates
against real labels does not protect against a steering harness picking up
a decoding-degeneracy artifact at high alpha. This is also the direct
precedent for the alpha-selection bug caught and fixed in
`docs/L52_LAYER_SUBSET_STEERING.md`: that fix (`SAFE_ALPHAS = (0.1, 0.25, 0.5)`,
restricting `best_alpha` selection) is exactly the guard that would have
caught this result automatically had it existed when L51 ran. Every
L53-L57 target script applies that fix from the start.

## Cost

Single script run, `plm_steering/l51_run_repro.py`, 5 alphas x 2 directions
x 150 eval sequences, Apple Silicon MPS. Raw scores/sequences saved to
`plm_steering/l51_repro_out/results.json`.
