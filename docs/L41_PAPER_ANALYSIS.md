# L41 — What the ESM-C Paper Actually Claims (read directly, 2026-07-21)

Read the full text of Candido, Hayes, Derry, ... Rives et al., "Language
Modeling Materializes a World Model of Protein Biology" (bioRxiv
10.64898/2026.06.03.729735, Biohub/EvolutionaryScale, June 2026) directly —
not via a research-agent summary — after the L41 normalization bug (see
docs/L41_PROTOCOL.md) made clear that secondhand summaries had already
caused one real methodological error. This doc separates what the paper
actually claims from what L41 tested, and states plainly where they diverge.

## What the paper actually claims (three separable claim clusters)

This is a large, multi-part systems paper, not a single-claim interpretability
study. Three claim clusters, in the paper's own order of emphasis:

**1. Scaling claim.** ESMC (a masked-LM, 300M/600M/6B params, trained on
~2.8B metagenomic sequences — ~56x more than ESM2's ~50M) shows log-linear
improvement in representation quality with training compute, matching or
beating ESM2 at a fraction of the parameters (ESMC-300M ≈ ESM2-650M on
contact precision; ESMC-6B beats the largest ESM2 models by a wide margin).

**2. Structure-prediction and design claim.** ESMFold2 (ESMC embeddings +
diffusion structure head) beats prior methods on biomolecular complex
prediction, including antibody-antigen interactions, and a search procedure
over this model finds nanomolar-affinity miniprotein/scFv binders with real
experimental (wet-lab) validation across five targets.

**3. Interpretability/organization claim — the one L41 actually tested.**
Using mechanistic-interpretability tooling (SAEs), the paper claims ESMC's
latent space organizes protein biology into a "reductionist" hierarchy of
concepts — from single-residue/secondary-structure features up to
family-specific and cross-lineage evolutionary themes — and that within this
organization, specific linear directions correspond to enzyme **function**
(EC number) in a way that is **independent of structure** (CATH fold),
verified via a structure-controlled benchmark (EC-CATH).

## Was claim #3 (the one L41 tested) actually achieved by the paper?

**Yes, but as a *discovery/probing* result, not a *causal/generative* one —
and L41's original framing ("the discovery paper stops short of steering")
was correct, but the paper's own validation methodology is far more careful
than what L41 replicated.** Specifics, read directly from the methods:

- **Benchmark:** EC-CATH — 5,829 enzyme-positive proteins vs. 9,211
  **structure-matched** negatives (same CATH fold, different EC number),
  73 leave-one-CATH-topology-out train/test splits across 32 EC numbers and
  42 topologies, with near-duplicate sequence pairs manually removed. This
  is a genuinely rigorous design specifically constructed to prove
  function-encoding survives holding structure constant — far more careful
  than a plain kinase-vs-random-negative split.
- **Method:** logistic/ridge regression (a **linear probe**, not raw SAE
  feature lookup) on **mean-pooled** per-layer dense embeddings, with the
  regularization strength and layer chosen per-task by cross-validation.
  Separately, SAE feature analysis (Figure 4) uses a *different* pooling
  convention — **max-pooling** across the sequence, not mean-pooling —
  specifically because SAE features are sparse and often fire only at a
  localized motif (e.g. a catalytic P-loop), which mean-pooling would dilute.
  **The SAE encoder's inputs are also Z-score normalized** before encoding
  (the exact step L41's original Gate 1 omitted — see docs/L41_PROTOCOL.md).
- **Scale:** the paper's headline SAE analysis (Figure 4, including the
  P-loop/kinase example) is run on **ESMC-6B, layer 60 of 60** (the
  penultimate layer) — chosen because it's empirically "near the peak" on
  the EC-classification task per a full layerwise sweep (Figure 1D). The
  6B model wins 53/73 (72.6%) of the EC-CATH tasks outright; smaller models
  win the remaining ~27%, so scale isn't a strict requirement for every
  task, but it's the dominant trend and the paper's own choice of "flagship"
  configuration.
- **What the paper does NOT do:** there is no causal intervention anywhere
  in the paper. Confirmed directly: zero occurrences of "causal," "ablation,"
  "intervention," or "perturb" in the full text; "clamp" and "inject" both
  appear only in unrelated contexts (a diffusion-module hyperparameter and
  physical sample injection for wet-lab chromatography, respectively). The
  paper's SAE section is a **discovery and probing** result — showing
  directions exist and correlate with EC-CATH-verified function — not a
  demonstration that adding a direction to activations *causes* generation
  to shift toward that function. L41's original framing of the gap was
  correct on this specific point.

## Did L41 actually test the paper's claim?

**Only loosely — L41 tested an analogous but meaningfully different, less
rigorous version of the claim, at a smaller scale, with two real methodology
gaps (both found and partially fixed mid-arc, see docs/L41_PROTOCOL.md):**

| | Paper (EC-CATH / Figure 4) | L41 (as run) |
|---|---|---|
| Model scale | ESMC-6B (headline), sweep across 300M/600M/6B | ESMC-300M only |
| Layer | 60 of 60 (empirically peak-validated) | 20 of 30 (arbitrary guess) |
| Negative class | Structure-matched (same CATH fold, different EC) | Arbitrary non-kinase UniProt sequences, no structural control |
| Pooling for SAE features | Max-pool across sequence | Mean-pool (fixed for the *dense*-embedding EC-CATH task in the paper, but L41 used mean-pool for its SAE-feature search too, diverging from the paper's own SAE-specific convention) |
| SAE input normalization | Z-score normalized (stated explicitly) | **Omitted in the original run** — found and fixed post-hoc; changed which feature "won" (7196 → 10004) |
| What was tested | Correlational: does a probe/feature separate function from structure | Causal: does adding the direction to activations *change generation* — genuinely novel, not something the paper attempted |

The corrected L41 result (after the normalization fix) is a **weak,
inconsistent-but-directionally-positive signal** (effect sizes 0.12–1.08
baseline-SE at n=60, not independently significant) — not a confirmed
causal effect, and not a repudiation of the paper's claim either, since the
paper never made a causal claim to begin with.

## How to actually test this properly

If the goal is a real, defensible test of "do these SAE function-directions
causally steer generation," the gaps above point to specific, concrete fixes,
roughly in order of expected impact:

1. **Use the paper's own validated configuration first, before generalizing.**
   Rerun Gate 1's feature search on **ESMC-6B, layer 60**, using the EC-CATH
   dataset's structure-matched negative sampling (same CATH topology,
   different EC), not an arbitrary non-kinase pool. This directly tests the
   paper's own best-validated setup rather than an analog at a smaller,
   less-tested scale. Practically: ESMC-6B requires more VRAM/time than fit
   the original 300M-scale plan (an A10G's 24GB is tight for 6B inference;
   likely needs a bigger GPU or careful batching/offloading) — this is the
   single biggest scope change, not a small tweak.
2. **Match the paper's SAE feature-aggregation convention exactly.** Use
   max-pooling (not mean-pooling) when building the per-sequence feature
   vector for the Cohen's-d search, since that's what the paper's own SAE
   analysis uses and mean-pooling systematically dilutes sparse, localized
   features.
3. **Structure-match the negative class.** Pull CATH/TED topology
   annotations for the negative (non-kinase) sequences and require them to
   share a fold with at least some kinase-family proteins, ruling out "the
   effect is just fold-recognition in disguise" as an explanation for
   whatever separation is found.
4. **Move from single-shot mask-fill to iterative refinement, or to an
   explicitly generative ESM (ESM3) rather than ESM-C.** ESM-C is masked-LM
   only; the single-shot mask-fill approximation used in L41's Gate 2 gives
   steering exactly one forward pass to compound its effect. Huang et al.'s
   ICML 2025 steering paper (the one directly-comparable prior causal-steering
   result, on ESM2/ESM3/ProLLaMA, not ESM-C) used proper generation loops,
   not single-shot fill — this is a likely reason L41's effect sizes stayed
   small even after the normalization fix.
5. **Increase sample size and add a pre-registered significance bar.** n=60
   sequences at one alpha sweep, no multiple-comparison correction, is
   underpowered to distinguish "real small effect" from "noise that happens
   to point the right way three times." A properly powered version would
   fix an effect-size target and required sample size *before* running,
   matching this project's usual gate-based discipline (see
   docs/L38_PROTOCOL.md, docs/L41_PROTOCOL.md).

None of this has been run — this is the honest scope of "what it would take,"
not a claim that it's been done. Given the ESMC-6B compute requirement, this
would be a materially larger and more expensive follow-up than the original
300M-scale L41 arc, not an afternoon's rerun.
