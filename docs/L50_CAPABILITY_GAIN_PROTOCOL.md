# L50 — Phase 0: Pre-Registered Definition of "Capability Gain"

**Pre-registered protocol.** Locked 2026-08-04, before any Phase 1+ run.

## Why this exists

L41-L49 were entirely diagnostic: every result answered "how does the model
represent X" or "is component Y causally load-bearing," never "here's a new
thing the model can now do." The one exception, L42's thermostability
steering, is a *reproduction* of Huang et al. 2025, not a new capability.

Before running Phase 1 (layer-targeted steering) or Phase 2 (composite
steering vectors) — the first attempts in this arc explicitly aimed at
capability, not diagnosis — the bar for "this is a real gain, not a
flattering number" needs to be locked in advance. This is the same
discipline that caught L42 v1's false PASS (poly-leucine collapse) and
L43's false significance (A/G collapse): decide the criteria before seeing
results, not after.

## Definition — a capability gain is real only if it clears ALL SIX:

1. **Beats both controls, head-to-head, with a real CI.** Real
   config/direction significantly beats (a) unsteered baseline AND (b) a
   matched-norm random-direction or matched-size random-group control —
   direct paired bootstrap between real and control, not two separate
   vs.-baseline tests. (Reused unchanged from L42/L43/L45/L47/L48.)
2. **Dose/scale-response, not a one-off.** A sweep parameter (alpha, K in a
   composite vector, number of refinement iterations) must show a coherent
   trend across at least 3 points, not one lucky value. (The exact check
   that caught L43's alpha=2.0 false positive.)
3. **Survives the residue/degeneracy exclusion check.** If the effect is
   driven by collapse into a narrow amino-acid composition, it's an
   artifact, not a capability — rerun scoring with the dominant substituted
   residue(s) excluded; effect must survive. (L42/L43's leucine/A-G check.)
4. **Independent scorer, validated against ground truth BEFORE the run, not
   after.** No proxy is used in a PASS decision without first being checked
   against real experimental labels. (The GRAVY lesson: r=-0.03 against real
   labels should have been checked before L43 ran, not discovered after.)
5. **Beats the current best-known technique, not just zero.** Since L42
   already gives a working thermostability baseline (all-33-layer,
   alpha=0.25), any Phase 1/2 claim of improvement must run head-to-head
   against that exact configuration, same eval set, same seed — not just
   "beats random."
6. **Adequately powered.** Minimum 30 non-degenerate pairs (existing guard,
   unchanged), but per L43's follow-up (which only resolved cleanly at
   n≈288), default to n=150+ eval sequences for any claim involving a new
   target property, not n=60.

**KILL:** any criterion fails outright → not a capability gain; log and move
to the next candidate. **AMBIGUOUS:** passes 1-2 but fails on 3 or 4 →
informative, not actionable (identical semantics to L43's original
AMBIGUOUS clause).

## Infra check 1 — layer-subset hook, self-identity/no-op

Before trusting Phase 1's layer-subset steering (hooking only layers
{18, 23, 25, 30, 31} — L45's confirmed causal-necessity set — instead of
all 33), verified the hook mechanism directly on ESM2-650M:

Registered a no-op-returning hook (`return output` unmodified) on each of
the 5 target layers and confirmed the hidden state each hook *captures*
exactly matches the corresponding entry of an un-hooked forward pass's
`output_hidden_states=True` output (max abs diff = 0.0 at all 5 layers).
Confirms hook registration/removal at these 5 specific layer indices is
correct and non-interfering when returning an unmodified value.

## Infra check 2 — layer-subset hook, real modification (with one dead-end documented)

**First attempt (dead end, documented rather than silently discarded):**
tried to verify the ACTUAL steering hook (`MultiLayerSteeringHook`, alpha>0)
by comparing `output_hidden_states=True` tensors before/after hooking layer
18. Result looked like a clean FAIL: hidden_states[19] (expected to reflect
layer 18's modified output) showed ZERO difference from baseline, while
hidden_states[20] showed a large, unexplained difference — inconsistent
with straightforward sequential layer indexing.

**Root cause, found by reading the actual transformers source directly**
(not assumed): this installed transformers version builds
`output_hidden_states` via a decorator (`capture_outputs` in
`transformers.utils.output_capturing`) that installs its OWN internal
forward hooks lazily, the first time `output_hidden_states=True` is
requested on a given model instance, and keeps them installed afterward.
Registering a second, user-added hook on the same layer and then requesting
`output_hidden_states=True` again does not reliably attribute the
user hook's modification to the right position in the resulting
`hidden_states` tuple — the internal capture hooks and the external
steering hook interact in an order that isn't the naive "each layer's own
hook fires in registration order" story. This is a real, if narrow,
interaction footgun in this transformers version, worth remembering for
any future check that mixes `output_hidden_states=True` with custom hooks
on the same model instance — but it does NOT affect the actual research
scripts, none of which combine the two (L42/L43/L45/L47's real generation
path, `mask_fill_generate`, never requests `output_hidden_states`).

**Correct verification: use the actual production code path.** Reran the
check via `mask_fill_generate` (the exact function every real experiment
calls) end to end:

| comparison | chars differing (of 78) | expected |
|---|---|---|
| baseline vs. 5-layer-subset steered (alpha=5) | 20 | > 0 |
| baseline vs. all-33-layer steered (alpha=5) | 23 | > 0 |
| 5-layer-subset vs. all-33-layer (both alpha=5) | 8 | > 0 (must NOT be identical) |
| baseline vs. 5-layer-subset at **alpha=0** | 0 | = 0 (true no-op) |

All four conditions hold. **The layer-subset hook is mechanically sound**:
it modifies generation, alpha=0 is an exact no-op, and restricting to 5
layers produces genuinely different (not identical) output from steering
all 33 — confirming the subset really is a strict, distinguishable subset
of the full intervention, not silently equivalent to it.

## Status

Phase 0 criteria locked; both infra checks pass (the second, after
resolving a real but narrow verification-methodology dead end, not a bug in
the hook itself). Phase 1 (layer-targeted steering at layers
{18, 23, 25, 30, 31}, head-to-head against L42's all-33-layer baseline) is
cleared to run against this protocol.

**Phase 1 complete — AMBIGUOUS, not PASS.** See
`docs/L52_LAYER_SUBSET_STEERING.md`: the 5-layer subset genuinely steers
thermostability (passes criteria 1-4, 6) but retains only ~43-59% of
all33's effect size at every alpha where both are trustworthy — a real,
significant, non-artifactual shortfall on criterion 5, not a wash. One
lesson for future Phase 1/2 runs: don't let `best_alpha`/comparison-alpha
selection range outside a technique's already-established safe operating
window (here, L42's own alpha < 1.0 constraint) — the first draft of L52
picked alpha=2.0 and got a spurious PASS purely from the two arms
collapsing into degenerate output at different alphas, not from a real
advantage. Caught before being recorded as a result; same discipline this
document exists to enforce, applied to itself.
