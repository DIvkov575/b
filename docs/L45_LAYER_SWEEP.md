# L45 — Per-Layer Causal Sufficiency Sweep of L42's Steering Vector

**Not pre-registered as a PASS/KILL gate** — this is a mechanistic follow-up
on an already-validated result (L42, docs/L42_STEERING_REPRO.md), not a new
claim requiring its own kill test. The question is "which layers are doing
the causal work," not "does steering work at all" (already answered: yes).

## Why this exists

L42 confirmed real, significant, dose-responsive difference-of-means
steering toward thermostability, applying the steering vector to all 33
transformer layers simultaneously. That leaves open which layers actually
matter — a prerequisite check before "does the reproduction generalize to a
new target" (L43, which came back AMBIGUOUS) is even the right next
question. This checks something more basic: is the effect concentrated in
a few causally-load-bearing layers, or a genuinely distributed property of
the whole residual stream?

An earlier same-day attempt at a different mechanistic question — whether
the over-steering collapse residue (leucine for L42, alanine/glycine for
L43) is predictable in advance via a logit-lens projection of the steering
vector through the model's embedding matrix — was tried and falsified
cleanly (src/l38/l44_logit_lens_diagnostic.py): after fixing two real bugs
in the check itself (routing through the wrong projection; getting fooled
by outlier embedding norms on rare/non-standard residue tokens), neither
target's steering vector showed the collapse residue as a dominant
single-layer direction. That's a real negative result, not a dead end
worth hiding — it means the collapse is a downstream decoding-dynamics
effect (aggregate pressure compounding across all layers during iterative
generation), not something legible in a static, single-layer linear
projection.

## Method

Reused L42's exact steering vectors, eval sequences, degeneracy filter,
IVYWREL scorer, and paired-bootstrap significance test unchanged
(`src/l38/l45_layer_sweep.py` imports directly from `l42_run_repro.py` and
`l42_steering_repro.py` rather than reimplementing). For each of the 33
layers independently: register the steering hook on THAT LAYER ONLY (not
all 33), generate all 60 held-out sequences at alpha=0.25 (L42's cleanest
non-collapsing signal), score, and compare against the same procedure using
a matched-norm random direction on that same single layer. `alpha=0.25`
chosen specifically because it's non-collapsing in the all-layers case —
a single layer is presumed even less likely to collapse generation on its
own, and this was confirmed (degenerate count stayed at 58/60 non-degenerate
pairs at every layer, identical to L42's own alpha=0.25 baseline).

## Results

**33/33 layers tested; 30 of 33 showed a positive real-vs-random effect
direction (binomial sign test p=0.000001 against the 50/50-chance null)**
— this is the headline finding, and it's robust to any single-layer
significance threshold, unlike the raw per-layer bootstrap CIs. 11 of 33
layers individually cleared a 95% CI on their own (layers 3, 9, 10, 18, 20,
22, 23, 24, 25, 30, 31) — far more than the ~1.65 expected by
multiple-comparisons chance at p=0.05 across 33 independent tests, so this
isn't just noise inflating a few CIs past zero.

**Effect size grows with layer depth, and this is NOT a vector-norm
renormalization artifact — checked directly.** Raw per-layer steering
vector norms grow ~8x from early to late layers (layer 0: 5.18, layer 31:
38.33), which could trivially explain a growing effect if the hook's
renormalization weren't doing its job. Checked directly: computed the
actual RELATIVE perturbation strength per layer
(`alpha * vector_norm / mean_hidden_state_norm`, since hidden-state norms
also grow ~8x across depth, layer 0: 96.5, layer 31: 761.6) — this relative
push stays roughly flat across all 33 layers (0.011–0.025, no depth trend)
and is NOT correlated with the observed effect size (Spearman
rho=-0.09, p=0.60). **The growing effect toward deep layers is a real
depth-dependent phenomenon, not an artifact of unequal perturbation
strength.**

**Layer 31 (second-to-last) stands out sharply**: effect size 0.00474,
roughly 3x the next-largest single layer (18 and 22, both 0.00161) and
~7x the mean of the other 32 layers. Layers 18, 20, 22-25, and 30 form a
secondary cluster of elevated (but smaller) effects in the back third of
the model. Layer 32 (the final layer) drops back near zero (0.00005) —
the ONE outlier layer whose own vector norm (1.04) and hidden-state norm
(10.5) are both far smaller than every other layer's, consistent with it
being a specialized final-representation layer rather than part of the
same causal chain as 0-31.

## Interpretation

This is consistent with — though does not on its own prove — a "later
layers do more of the causally load-bearing computation for this specific
property" story, which would be a genuinely novel, checkable mechanistic
claim about how thermostability-relevant information is organized in
ESM2-650M's residual stream. It does NOT show the effect is localized to
one layer (30/33 positive, not just a small significant cluster) — the
correct characterization is "distributed but depth-weighted," not
"concentrated in 2-3 special layers."

## What this is NOT

Not a claim that layers 18-31 are the "thermostability circuit" in any
strong mechanistic-interpretability sense (that would require ablation
studies, attention-pattern analysis, or activation patching between
matched clean/counterfactual pairs — none of which was done here). This is
a causal-SUFFICIENCY sweep (does steering this layer alone reproduce part
of the effect), not a causal-NECESSITY sweep (does removing/ablating this
layer's normal computation destroy the all-layers effect) — the two ask
different questions and only sufficiency was tested.

## Cost

Reused L42's exact model, data, and vectors. Runtime: two full 60-sequence
generation passes (real + random control) per layer, 33 layers, on Apple
Silicon MPS: single 8-minute run, no GPU cluster needed. Cheap enough that
running it again with a different alpha or on L43's solubility vectors
(to check whether the same depth-weighting pattern holds for a target
where the all-layers effect turned out to be an artifact) would cost about
the same again.
