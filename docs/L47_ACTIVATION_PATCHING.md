# L47 — Activation Patching: Feasibility, Task B Validation, Task A (Vig Redo)

## Plan (as proposed and approved 2026-07-31)

A dedicated literature check found that no paper has taken Vig et al.'s
2021 "BERTology Meets Biology" (ICLR 2021, arXiv:2006.15222) finding —
specific attention heads in protein LMs whose attention weights strongly
correlate with contact maps / binding sites — and actually tested it
causally (ablate the head, measure whether contact/binding prediction
degrades). The authors' own paper states plainly: "all of the above
analyses are purely associative and do not attempt to establish a causal
link." That's the real, unclaimed gap this L47 arc targets.

Plan, phased:
- **Phase 0** — feasibility: can we cleanly intervene on a single attention
  head's output in HuggingFace's ESM2 implementation?
- **Phase 1** — pick tasks: **Task B** (cheap validation — patch L42's
  already-trusted thermostability harness with true substitution instead
  of additive steering, confirm the method itself works before trusting it
  on anything novel) then **Task A** (the actual novel contribution — redo
  Vig's contact/binding-site heads as a real causal test).
- **Phase 2** — build one generic, reusable patching harness.
- **Phase 3** — run the sweep(s).
- **Phase 4** — cross-check against everything already built (L45's
  steering-based depth-weighting, Vig's correlational heads).

## Phase 0: feasibility — CONFIRMED

Checked `transformers`' `EsmSelfAttention.forward()` directly: it computes
per-head outputs internally (`[batch, seq, num_heads, head_dim]`) but
merges them via `.reshape(*input_shape, -1)` before returning a single
`[batch, seq, hidden_size]` tensor. Verified directly (not assumed) that
this reshape is a plain view over head-ordered memory: captured a real
hook's output on ESM2-650M, reshaped it back to
`[batch, seq, num_heads=20, head_dim=64]`, and confirmed per-head slices
are exactly recoverable. **A single `register_forward_hook` on
`EsmSelfAttention` is sufficient for head-level patching** — no need for
`output_attentions=True` or deeper surgery.

Three additional sanity checks run directly on ESM2-650M before trusting
the harness (`src/l38/l47_activation_patching.py`):
1. Cross-sequence head patching produces a real, nonzero logit shift.
2. Patching two DIFFERENT heads (3 vs. 7) produces different-magnitude
   effects — confirms the head-slicing genuinely isolates one head, not
   silently patching the whole tensor.
3. Self-patching (patching a sequence with its OWN cached activation) is an
   exact no-op (max abs diff = 0.0) — confirms the mechanism is
   substitution done correctly, not a subtly-broken identity operation.

## Phase 1-3, Task B: validate patching against L42's trusted result

**Real bug found and fixed before the real run** (documented per this
project's discipline, not hidden): the first design broadcast a single
mean-pooled activation to EVERY token position at a layer, discarding all
per-token variation across the whole sequence. A smoke test at N=5 caught
this immediately: 100% of generated sequences degenerated at EVERY layer
tested, regardless of which vector was patched in. Root cause: overwriting
the context the model reads FROM (not just what it's predicting) destroys
its ability to produce anything coherent. Fixed by restricting the patch to
ONLY the masked positions actually being predicted — leaving all real
context untouched. Re-smoke-tested at N=5: non-degenerate generation
recovered through most of the network (only degrading near the final
layers, which is itself informative — see below).

**Method:** `MaskedPositionPatchHook` (src/l38/l47_task_b_patching_validation.py)
replaces a layer's output ONLY at the positions being masked/predicted,
with either the high-Tm group's mean-pooled activation (real condition) or
the low-Tm group's own mean-pooled activation (control — if this produced
the same effect as the high-Tm patch, the effect wouldn't be specific to
the high-Tm signal). Reused L42's exact data groups, degeneracy filter,
IVYWREL scorer, and paired-bootstrap significance test.

**Result:** 27 of 31 layers with a valid non-degenerate comparison show a
positive effect (high-Tm patch beats low-Tm patch), sign test
**p=0.000034** against the 50/50-chance null. 12 layers individually clear
a 95% CI (3, 5, 13, 14, 23-30). Effect size grows sharply with depth: mean
effect 0.0006 in layers 0-10, 0.0009 in layers 11-21, **0.0076 in layers
22-30 — a ~13x jump**, closely mirroring L45's steering-based finding
(effect tripled from early to late layers there). Two genuinely independent
causal methods — additive steering (L45) and substitutive patching
(L47 Task B) — converge on the same structural conclusion: **later layers
carry more of the causally load-bearing signal for thermostability.**

**Honest caveat, checked directly rather than glossed over:** layer 30
shows a SIGNIFICANT NEGATIVE effect (-0.0082), the only layer where the
sign flips. Checked whether this is a real reversal or an artifact: at
layer 30, high-Tm-patched generation degenerates (homopolymer collapse,
same failure mode as L42/L43) at more than double the rate of the low-Tm
control (15/60 vs. 7/60 degenerate, even after the degeneracy filter
already excluded the worst cases). Layers 31-32 show the same pattern
escalating (33/60 and 60/60 degenerate) until excluded entirely by the
`MIN_NONDEGENERATE_PAIRS` guard. **This looks like the same collapse
artifact documented in L42/L43, not a genuine sign reversal of the causal
effect** — deep-layer patching with a strong (mean-pooled, non-alpha-scaled)
substitution pushes generation toward the same degenerate failure mode,
and IVYWREL's scoring quirks on that failure mode likely explain the
apparent negative sign, the same way leucine-collapse fooled two different
metrics in L42. Not fully resolved — flagged honestly rather than either
ignored or overclaimed as a new finding.

## Phase 4 (partial): cross-check against L45

L45 (additive steering, single-layer causal sufficiency) found significant
layers at 3, 9, 10, 18, 20, 22-25, 30, 31, with the single strongest effect
at layer 31. L47 Task B (substitutive patching, masked-position only) finds
significant layers at 3, 5, 13, 14, 23-30, with the single strongest effect
at layer 28. **Real overlap, not identical**: both methods agree layer 3 is
significant and both find layers in the low-20s through 30 are the
strongest cluster — genuinely converging on "the back third of the network
matters more" via two structurally different causal interventions. They
disagree on the exact peak layer (31 for steering, 28 for patching) and on
some individual early/mid layers (5, 13, 14 significant in patching but not
steering, and vice versa for 9, 10, 18, 20, 22, 25). Given L45 itself found
30/33 layers positive (broadly distributed, not sharply localized), some
disagreement on which INDIVIDUAL layers cross a significance threshold is
expected even for a real, robust effect — the AGGREGATE pattern (depth-
weighted, back-third-dominant) is what both methods actually agree on.

## Status: Task A (Vig redo) not yet started

Task B validates the harness works and gives a real, converging finding on
its own. Task A — actually redoing Vig et al.'s contact/binding-site heads
as a causal ablation test on a real structure-prediction task — is the
next step, not yet built.
