# L52 — Phase 1: Does the Causally-Necessary 5-Layer Subset Preserve L42's Steering Effect?

**Pre-registered against `docs/L50_CAPABILITY_GAIN_PROTOCOL.md`'s 6 criteria**,
locked before this run. This is Phase 1 from that protocol: the first
capability-oriented (not purely diagnostic) experiment in the L41-L52 arc.

## Why this exists

L45's two independent sweeps (sufficiency and leave-one-out necessity) both
converged on the same 5 layers — **18, 23, 25, 30, 31** — as the causally
load-bearing subset of L42's all-33-layer thermostability steering vector.
That's a mechanistic finding. The capability question it enables: if you
only ever intervene on those 5 layers (leaving the other 28 completely
untouched), do you keep the steering effect L42 established, or was the
"necessity" signal a same-magnitude-either-way artifact of how the sign
test was scored? Phase 0 (`docs/L50_CAPABILITY_GAIN_PROTOCOL.md`) verified
the layer-subset hook itself is mechanically correct (real modification,
exact no-op at alpha=0, genuinely different output from the all-33 case)
before this run tests whether it's *usefully* correct.

## Method

`src/l38/l52_layer_subset_causal_steering.py` reuses L42's exact data
loading, vector-building split, eval-sequence selection (same `SEED=0`,
same 20/80 percentile split, same 150-per-group vector pool, same 60
held-out low-Tm eval sequences), degeneracy filter, IVYWREL proxy, and
paired-bootstrap significance test — importing directly from
`l42_steering_repro.py` rather than reimplementing. Two configs run
side by side in the same script invocation, so both see identical
steering vectors, random-control vectors, and eval sequences:

- **`all33`**: steering hook registered on all 33 transformer layers
  (L42's original configuration, recomputed fresh in this run rather than
  reusing L42's cached numbers, since the same-run/same-eval-set guarantee
  matters for a head-to-head claim).
- **`subset5`**: steering hook registered ONLY on layers {18, 23, 25, 30,
  31}, using the SAME per-layer vectors as `all33` (each layer's own
  difference-of-means direction), just not applied outside those 5.

Both run at the same 5-point alpha grid as L42 (`[0.1, 0.25, 0.5, 1.0,
2.0]`), against a matched-norm random-direction control, real-vs-random
compared via paired bootstrap (10,000 resamples), degenerate sequences
excluded before scoring.

**Critical correction made mid-analysis, not silently absorbed:** the
first draft of this script picked `best_alpha` (the value used for
criterion 3's residue-robustness check and criterion 5's head-to-head
comparison) from *any* alpha with a significant subset5-vs-random effect,
including alpha=1.0/2.0. That produced a spurious `"PASS"`: at alpha=2.0,
`all33` has **fully collapsed** into degenerate poly-leucine output
(0/60 non-degenerate pairs — the exact failure mode L42's own doc already
documents and explicitly quarantines, `docs/L42_STEERING_REPRO.md`'s
"Honest verdict" section restricting trustworthy comparisons to alpha in
[0.1, 0.5]), while `subset5` hadn't collapsed yet at that alpha. subset5
"won" the head-to-head at alpha=2.0 only because it breaks down later than
all33, not because it steers harder in any comparable regime — an
artifact of collapse-order, not a real advantage. Fixed by restricting
`best_alpha` selection to `SAFE_ALPHAS = (0.1, 0.25, 0.5)`, matching L42's
own established safe operating range, and rerunning end to end (not
patching the already-computed numbers).

## Results

| alpha | subset5 vs random | all33 vs random | subset5 vs all33 (head-to-head) |
|---|---|---|---|
| 0.1 | +0.0041, n=58, **sig** | +0.0069, n=58, **sig** | −0.0041, n=58, **sig worse** |
| 0.25 | +0.0096, n=58, **sig** | +0.0224, n=58, **sig** | −0.0150, n=58, **sig worse** |
| 0.5 | +0.0236, n=58, **sig** | +0.0498, n=57, **sig** | −0.0305, n=57, **sig worse** |
| 1.0 | +0.0463, n=58, sig (but all33 side collapsed) | untestable, 5/60 non-degenerate | untestable |
| 2.0 | +0.0749, n=57, sig (but all33 side collapsed) | untestable, 0/60 non-degenerate | untestable |

**subset5 genuinely steers thermostability** — real vs. random control is
significant at every alpha, with a clean monotonic dose-response
(+0.004 → +0.010 → +0.024 as alpha increases through the safe range), and
survives the residue-exclusion robustness check: excluding the two most
substituted residues (E, R) at the best safe alpha (0.5), the effect drops
from +0.024 to +0.011 but remains significant (CI [0.009, 0.014]) — the
same "shrinks but doesn't vanish" pattern L42's leucine-exclusion check
used to rule out pure-composition-collapse as the whole story.

**But it is NOT equivalent to steering all 33 layers.** At every alpha in
the safe [0.1, 0.5] range where both configs are trustworthy (non-collapsed
on both sides), subset5's effect is significantly SMALLER than all33's —
retaining roughly **43-59% of the full effect** (0.0041/0.0069=0.59 at
alpha=0.1; 0.0096/0.0224=0.43 at alpha=0.25; 0.0236/0.0498=0.47 at
alpha=0.5). This ratio is roughly stable across the safe alpha range, not
drifting toward parity as alpha increases, so it isn't simply "subset5
needs a bit more alpha to catch up" — the two are on different effect
curves, not offset versions of the same curve.

## Verdict: AMBIGUOUS (5 of 6 criteria pass; fails criterion 5)

| criterion | result |
|---|---|
| 1. beats both controls, real CI | **PASS** — significant at every alpha vs. matched-norm random |
| 2. dose/scale-response | **PASS** — monotonic increase across all 3 safe-range alphas |
| 3. survives residue-exclusion | **PASS** — effect survives with top-2 substituted residues excluded |
| 4. proxy pre-validated | **PASS** — inherited from L42 (IVYWREL, not re-derived here) |
| 5. beats/matches best-known technique | **FAIL** — significantly worse than all33 at every comparable alpha, not just "not better" |
| 6. adequately powered | **PASS** — n=57-58, same guard as L42 |

Per L50's own semantics ("AMBIGUOUS: passes 1-2 but fails on 3 or 4" was
written anticipating failure on the artifact-detection criteria
specifically; here the failure is on 5, a comparison criterion, which the
protocol didn't originally split out as its own KILL-vs-AMBIGUOUS case).
Treating it as AMBIGUOUS rather than KILL is the right call: this is not a
false positive like L42 v1/L43/L51 — the layer-subset effect is real,
survives every artifact check that catches those — it is simply a smaller
real effect than the full-layer version, which is informative (about how
much of the causal work is concentrated in the "necessary" 5 layers vs.
distributed across the rest) rather than a dead end.

## Interpretation

L45's necessity sweep already showed 27/33 layers have a *nonzero* positive
necessity signal when excluded one at a time (not just the top 5) — this
result is consistent with, not contradicted by, that finding: the top-5
layers matter MOST (largest single-layer drops when excluded), but the
remaining ~half of the effect is spread thinly enough across the other 28
layers that restricting to only the top 5 leaves real effect on the table.
"Causally necessary" (removing it hurts) and "causally sufficient on its
own to fully replace the whole" are different, and this is the concrete
capability-relevant gap between them: a 5-layer-only intervention is
cheaper and still real, but is a strictly weaker tool than the full
33-layer version for this specific property, not a free lunch.

## What this is NOT

Not evidence that a *different* choice of 5 layers, a larger subset (e.g.
top 10), or a re-scaled single-layer-equivalent alpha would fail the same
way — none of those were tested here. Also not a claim about any property
other than thermostability; L45's separate necessity-sweep negative control
already showed solubility's necessity ranking uses a different, non-
overlapping layer set, so this specific 43-59%-retention number is not
expected to generalize to L43/L51's killed targets even if their layer
subsets were retested here.

## Cost

Single script run, both configs (all33 + subset5) x 5 alphas x 2 directions
(real + random) x 60 eval sequences, on Apple Silicon MPS: ~4 minutes wall
clock for the full run (embedding + all 20 generation arms + bootstrap).
Raw per-arm scores and generated sequences saved to
`src/l38/l52_repro_out/results.json` alongside the verdict, so the head-to-
head or robustness numbers can be recomputed without rerunning the model.
