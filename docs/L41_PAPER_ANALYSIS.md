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

## Fresh literature check (2026-07-21) — is discrete-function steering at small scale a known-viable technique at all?

Ran a dedicated, from-scratch literature scan (not reusing the earlier thin
summary) specifically on: has anyone, anywhere, published activation
steering of a protein LM toward a *discrete enzyme function* (e.g. kinase
activity / a single EC number) using a *sub-1B model* and a *single
direction* (SAE feature or difference-of-means)? Full citations below.

**Finding: no. This exact combination has never been shown to work by
anyone, and three independent, already-published findings explain why a
first attempt would plausibly fail:**

1. **Discrete function vs. continuous property.** Every published
   single-direction steering success (Huang et al. 2509.07983, ICML 2025)
   targets a smooth biophysical scalar — thermostability, solubility, GFP
   brightness — never a discrete catalytic-function class. The one paper
   that steered toward a function-like discrete concept (ProtSAE,
   arXiv:2509.05309, "DNA-binding transcription repressor activity") needed
   a **15B-parameter model** and a specially disentangled/hierarchical SAE,
   not a raw direction, and still only reached partial (TM-score ~0.83)
   similarity to the target concept — not a clean causal win.
2. **Documented failure at almost exactly L41's scale.** ProGenMech/Circuit
   Tracing (arXiv:2606.16044, ICML 2026 workshop) attempted causal steering
   of a functional-fitness circuit on **ProGen3-112M** (smaller than but
   comparable in class to L41's 300M) and found **no measurable effect** —
   steered output was "distributionally indistinguishable" from baseline.
   The authors explicitly attribute this to model scale and state they
   expect it to work only after scaling up.
3. **Raw SAE features are explicitly flagged as unreliable for causal
   control**, independent of scale. The antibody-SAE paper (arXiv:2512.05794)
   states directly: "high feature-concept correlation does not guarantee
   causal control over generation" — only hierarchically-structured
   ("Ordered") SAEs reliably produced steerable features in their tests.
   L41 used a raw, off-the-shelf SAE decoder row, exactly the feature type
   flagged as unreliable.

Also confirmed: **no published or preprinted work combines ESM-C (the
paper L41 was built on) with causal activation steering at all**, as of
this search (checked the paper's own citation graph — 8 citing works, none
perform steering). This part of the gap is real and still open. But "open"
here means "nobody has gotten around to a harder, currently-unfavored
combination" — not "a promising, under-explored opportunity."

**Revised bottom line: L41's weak/null result is not an anomaly needing
further investigation — it lands exactly where three independent,
already-published findings predict a first attempt at this specific
combination (small model + single raw direction + discrete function) would
land.** A positive result would have been the surprising outcome. This is
not a promising direction to keep pushing on with the current setup; the
literature's one repeatedly-validated recipe for single-direction steering
success is a continuous biophysical property (stability/solubility) on a
mid-size model (viable candidates in the 650M-3B range), not a discrete
enzyme function on a 300M model.

Key sources: Huang et al. 2509.07983 (ICML 2025); ProtSAE 2509.05309 (AAAI
2026); ProGenMech/Circuit Tracing 2606.16044 (ICML 2026 workshop); Antibody
SAE steering 2512.05794; InterPLM 2412.12101; ESMC-SAE enzyme prediction
2606.12209 (discriminative only, no steering).
