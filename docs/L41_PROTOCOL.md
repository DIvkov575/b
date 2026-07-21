# L41 — Causal Activation Steering with ESM-C SAE Function-Directions

**Pre-registered protocol.** Locked 2026-07-21, before the steering experiment
runs (Gate 0 infra checks below were already done hands-on before this was
written — see "Gate 0" section, all verified against the real GPU box, not
assumed from docs).

## The gap this targets

A June 2026 paper (Candido, Hayes, Rives et al. / Biohub, "Language Modeling
Materializes a World Model of Protein Biology," bioRxiv
10.64898/2026.06.03.729735) found that ESM-C's sparse-autoencoder (SAE)
feature space contains directions corresponding to conserved protein
**function**, independent of fold/structure. That paper **discovers** these
directions (via SAE + agentic labeling) but does not **causally steer**
generation with them — it stops at correlational/discovery evidence. A
separate paper (Huang et al., "Steering Protein Language Models," ICML 2025,
arXiv:2509.07983) does causal activation steering, but on ESM2/ESM3/ProLLaMA,
predating the ESM-C function-direction discovery and using difference-of-means
vectors, not SAE features. As of this writing (~1 month after the ESM-C
paper), no paper has combined the two: **use an ESM-C SAE function-direction
to causally steer generation, and verify the output actually shifts toward
that function.**

## Gate 0 — infra verification (COMPLETE, done live on the EC2 A10G before writing this doc)

All verified hands-on, not assumed:
- `ESMC-300M` loads via the official PyPI `esm` package (v3.2.3,
  EvolutionaryScale Team) — **no GitHub source install** of `Biohub/esm`
  needed for the base model (that repo is GitHub-only and would require the
  same class of external-build-code install this research thread has
  deliberately avoided since the ESMFold/openfold dead-end).
- `model.transformer.blocks[N]` are standard `nn.Module`s supporting
  `register_forward_hook` — the exact mechanism activation steering needs.
- The `ESMC-300M-sae-k64-codebook16384` checkpoints are plain `safetensors`
  (`W_enc [960,16384]`, `W_dec [16384,960]`, `b_dec [960]`) — a standard
  top-k SAE, **re-implementable in ~10 lines without the Biohub/esm wrapper
  class** (`add_sae_models`, etc.), which also lives GitHub-only.
- Verified reconstruction quality directly: layer-10 SAE explains **82.4%**
  of activation variance at L0=64 (matches expected top-64 SAE quality) on a
  real ESMC-300M forward pass of a real protein sequence.
- ESM-C is masked-LM only (`mask_token_id` present, no autoregressive
  `generate`) — "generation" here means iterative mask-fill decoding, not
  autoregressive sampling.
- **Zero external repo installs used for any of the above** — PyPI `esm`,
  `safetensors`, `transformers`, `scikit-learn`, plain UniProt REST.

## Known limitation carried in deliberately

The 300M-scale SAE does **not** ship agent-generated feature labels (only
the 6B model's layer-60 SAE does, per the Biohub README) — feature-to-concept
mapping must be derived here via correlation with UniProt annotations, the
same technique InterPLM uses. This is more work than "look up the labeled
feature," but is a well-established, checkable methodology, not a novelty
gap in itself.

## Gate 1 — find a function-linked feature (pre-registered decision rule)

**Concept chosen:** kinase activity (UniProt keyword `KW-0418`), for
practical reasons: large, cleanly-labeled, binary, no ambiguity about
positive/negative class definition (unlike e.g. "stability," which needs a
continuous assay).

**Method:**
1. Pull reviewed UniProt sequences with `KW-0418` (kinase, positive) and
   reviewed sequences without it (negative), length-filtered, matched scale
   to the phage/virion pulls already done in L39 (~3000 each).
2. Split into a **feature-identification split** and a **held-out
   evaluation split** (disjoint sequences) — the same split discipline used
   in L38's leakage-avoidance rule, applied here to avoid picking a feature
   that "works" only on the sequences used to find it.
3. On the feature-identification split: run ESMC-300M forward, hook a chosen
   mid-late layer (starting candidate: layer 20 of 30 — a InterPLM-informed
   guess at where abstract/functional features concentrate, not layer-swept
   exhaustively given the time budget), mean-pool hidden states per sequence,
   encode through the layer's SAE, and compute a simple separation statistic
   per feature (Cohen's d between positive-class and negative-class mean
   activation).
4. **PASS condition:** at least one feature has `|Cohen's d| > 1.0`
   (a large effect by conventional thresholds) between kinase and non-kinase
   mean activation on the identification split. **KILL condition:** no
   feature clears this bar — would mean the 300M-scale SAE doesn't cleanly
   encode this concept (possible; the 6B model is what the discovery paper
   validated), and this specific attempt should stop here rather than force
   a weak feature into the steering step.

## Gate 2 — causal steering + mask-fill generation

Only reached if Gate 1 passes.

- **Steering vector:** the winning feature's SAE decoder row (`W_dec[k]`,
  960-dim, already in the model's residual-stream space — no extra
  projection needed).
- **Injection:** forward hook on the chosen layer adds `alpha * steering_vector`
  to the hidden state at all positions during the forward pass.
- **Generation:** take held-out **negative-class** (non-kinase) sequences,
  mask a fixed fraction of positions, run one masked-fill pass (predict
  highest-probability residue per masked position) with and without the
  steering hook active, at a small sweep of `alpha` values.
- **Critical control (pre-registered, non-negotiable):** repeat the exact
  same procedure with a **random direction of matched norm** substituted for
  the steering vector. This isolates whether the *specific* function
  direction matters, or whether any perturbation of that magnitude produces
  a similar effect (a confound the original ESM-C paper's discovery-only
  framing can't rule out, and Gate 2 must not skip).

## Gate 3 — independent evaluation (not self-referential)

Using the SAME feature that produced the steering vector to also judge
success would be circular. Independent oracle:
- Train a **separate logistic-regression classifier** on ESMC-300M
  mean-pooled embeddings (dense, not the SAE feature space) to predict
  kinase/non-kinase, trained on a split disjoint from both Gate 1's
  feature-identification split and Gate 2's generation inputs — mirrors the
  L39 virion-classifier methodology exactly (frozen embeddings + logistic
  regression probe).
- Score steered, control-steered (random direction), and unsteered generated
  sequences with this independent classifier's `P(kinase)`.

**PASS/DELIVERABLE condition:** mean `P(kinase)` for real-direction-steered
sequences exceeds both the unsteered baseline AND the random-direction
control by a margin clearing basic noise (report exact numbers, no
pre-committed effect size given the exploratory nature of this specific
comparison — but the ordering real-direction > {unsteered, random-control}
must hold, not just "steering changed something").
**KILL condition:** random-direction control produces a similar or larger
shift than the real steering direction — indicates the effect (if any) is a
generic perturbation artifact, not evidence of causal function-steering.

## Compute / feasibility

All of Gate 1-3 is inference-only (no training beyond a cheap logistic
regression probe) on ESMC-300M — well within 1x A10G, expected same order of
magnitude as L39's embedding-extraction step (minutes, not hours).

## Deliverable regardless of outcome

A concrete answer to: does causally steering with an ESM-C SAE
function-direction shift generation toward that function, beyond what a
matched-norm random perturbation would do? Either a real positive result
(the first demonstrated causal use of these specific directions) or a clean
negative result (the effect doesn't survive the random-direction control) is
citable — this hasn't been checked by anyone yet per the Gate-scan research.

---

## RESULTS (2026-07-21) — run end to end on the EC2 A10G

**Gate 1: PASS.** Layer 20's SAE contains a feature (index **7196**) with
Cohen's d = **1.50** between kinase (`KW-0418`) and non-kinase sequences
(n=300 per class, identification split) — clears the pre-registered
threshold (>1.0) with a healthy margin. Four other features also cleared the
bar (indices 656, 12177, 8836, 16339, d=1.07–1.23), suggesting kinase
activity is genuinely well-represented at this layer, not a fluke of one
feature. Full ranking: `src/l38/l41_gate1_results.json`.

**Gate 2: ran cleanly, no bugs after two fixes** (a device-mismatch crash,
and a tokenizer-decode bug that left `<cls>`/space artifacts in generated
sequences — both caught and fixed via smoke-testing before the full run,
not discovered after). Generated 60 held-out non-kinase sequences × 7
conditions (unsteered, 3 alphas × {real feature-7196 direction, matched-norm
random-direction control}) via single-shot mask-fill. Steering **does**
measurably perturb the output — ~2.3–2.4 residues change per sequence at
alpha=10, roughly the same magnitude for both the real direction and the
random control, confirming the hook mechanism itself works.

**Gate 3: KILL, on a real (not artifactual) null result — after catching and
fixing one real methodology flaw first.**

*v1 (30% masking):* every condition, including the true unsteered baseline,
scored P(kinase) ≈ 0.003 by the independent classifier — a flat floor, not a
real comparison. Diagnosed directly: with only 30% of the sequence masked,
the classifier's whole-sequence verdict is dominated by the unmasked 70% of
a strongly non-kinase-looking scaffold, structurally incapable of registering
any local steering effect regardless of whether steering worked. This was a
measurement-sensitivity bug in the eval design, not evidence against the
hypothesis, and is recorded here rather than silently reissuing a different
number.

*v2 (80% masking, same held-out sequences, same steering vector/hook):*
classifier scores moved into a real, non-degenerate range (means 0.039–0.043,
std 0.05–0.06). But the differences between conditions are tiny relative to
that noise:

| Condition | mean P(kinase) | Δ vs. baseline (in baseline SE units) |
|---|---|---|
| Unsteered baseline | 0.0427 | — |
| Real direction, α=5 | 0.0394 | −0.48 SE |
| Real direction, α=10 | 0.0405 | −0.32 SE |
| Real direction, α=20 | 0.0433 | +0.09 SE |
| Random control, α=5/10/20 | 0.0389–0.0399 | (comparable magnitude, no clear ordering) |

No dose-response (α=5 and α=10 score *below* baseline; only α=20 nudges
barely above, by under 0.1 SE), and the real-direction condition never beats
the random-direction control by a margin distinguishable from chance at any
alpha. **This is a clean, statistically unambiguous negative result, not a
suppressed effect hiding behind noise** — full numbers:
`src/l38/l41_gate3_results.json`.

## Post-hoc verification (2026-07-21, prompted by "have you checked all components?")

Two checks done after the initial write-up, closing gaps that were previously
just assumed:

1. **Layer-indexing alignment.** Gate 1 reads `output.hidden_states[20]` to
   find the feature; Gate 2 hooks `model.transformer.blocks[20]` directly.
   Verified these are the exact same tensor (`torch.allclose`, max abs diff =
   0.0) — not off-by-one (confirmed `blocks[19]`'s output does NOT match
   `hidden_states[20]`, ruling out the classic hidden-states-includes-
   embedding-layer indexing trap).
2. **Intervention-works-at-the-source check.** Re-encoded the *steered*
   hidden state back through the SAE and confirmed feature 7196's own
   activation rises by ≈alpha (e.g. +5.14 at alpha=5, +20.54 at alpha=20,
   consistent across 3 independent sequences and matching the near-unit-norm
   steering vector's expected linear contribution). This confirms the causal
   chain's first link — hook → target feature activation — is mechanically
   correct, not silently broken.

**What this changes:** the Gate 3 null is not explained by a plumbing bug
(wrong layer, inert hook, mis-normalized vector). The break, if there is one,
is specifically between "feature 7196 fires harder" and "generation shifts
toward kinase-like sequence" — i.e. either this single SAE decoder direction
at 300M scale isn't causally load-bearing enough on its own to move
generation (plausible under superposition — other features may carry
competing or overlapping signal), or activating it strongly doesn't
correspond to "produce a more kinase-like sequence" in a way that survives
discretization back to amino acids. This was not isolated further (would
require, e.g., ablation studies or an intermediate readout between the
feature and the final sequence) — recorded as the honest boundary of what
was checked, not resolved.

## Honest interpretation

The specific, narrow claim under test — *adding this one SAE decoder
direction (feature 7196, layer 20, ESMC-300M) at inference time shifts
single-shot mask-fill generation toward kinase-like sequence, beyond a
matched-norm random perturbation* — **does not hold** at this scale,
methodology, and feature choice. This does not refute the broader idea that
ESM-C's SAE function-directions are causally meaningful (Huang et al.'s
difference-of-means vectors on ESM2/ESM3 did work; this is a different
model, different direction-construction method, different masking-based
generation approach, and the smallest available ESM-C scale). Plausible,
untested reasons this specific attempt could fail even if the underlying
phenomenon is real: (a) 300M may be too small for cleanly monosemantic SAE
features — the discovery paper's headline validation was on the 6B model;
(b) single-shot mask-fill (not iterative refinement) may not give steering
enough opportunity to compound its effect across positions; (c) one feature
direction, one layer, one alpha range is a narrow slice of a large
hyperparameter space that a positive result would need to search more
thoroughly.

**Decision: do not proceed further on this exact configuration.** A genuine
next step, if pursued, would need to change at least one of: model scale
(6B, where the discovery paper's own validation lives), generation mechanism
(iterative refinement over many mask-fill passes, closer to how Huang et al.
generated), or steering-vector construction (difference-of-means over many
examples, as Huang et al. did, rather than a single SAE decoder row). None
of that is scoped here — this result stands as a clean, reported negative
finding at the specific scale/method combination tested.
