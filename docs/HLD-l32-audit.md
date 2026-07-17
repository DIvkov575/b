# HLD: Silent-Degradation Audit of Few-Step Protein Generators (`l32-audit`)

**Date:** 2026-07-12 · **Author:** divkov · **Status:** Draft

## 1. Overview

### 1.1 Background
- As of mid-2026 a wave of methods made protein backbone / structure generators *fast* by
  cutting the number of function evaluations (NFE) from ~50 to a handful: SiD-Protein
  (arXiv:2510.03095, >20×), Riemannian MeanFlow (arXiv:2602.07744, ~10×), Generalised Flow
  Maps (arXiv:2510.21608), plus folding-side few-step samplers (DeCAF arXiv:2606.08375,
  DCFold arXiv:2605.17899, Protenix-Mini arXiv:2507.11839 / 2510.12842).
- Every one of these reports the same headline: a scalar **designability** number
  (generate → ProteinMPNN → ESMFold → self-consistency RMSD < 2 Å) at a chosen NFE, usually
  at a single length (~300 residues), single oracle (ESMFold), single seed, no error bars.
- This repo's research thread spent three probes trying to *build a faster method* and found
  the space is a red ocean (`docs/RESEARCH_LEDGER.md`). A re-audit of the 30 killed candidates
  (2026-07-12) surfaced five that were killed on invalid or flawed grounds — two on citations
  that turned out to be the wrong paper — and they share one shape: **diagnostics of how the
  fast generators silently degrade**, not new fast generators.

### 1.2 Problem Statement
The field's efficiency claims rest on a metric that cannot see several distinct failure modes.
Five specific gaps, each verified open this session:
- **Mean-structure collapse (L18):** distillation losses use coordinate/frame distance; no
  per-sample, designability-correlated *perceptual* distance exists (Protein FID arXiv:2505.08041
  is distributional + eval-only, validated vs CATH/Foldseek, not designability).
- **Coverage collapse (L27):** Protein FID reports only a single Fréchet scalar — no
  precision/recall split, which is what exposes mode collapse.
- **Oracle dependence (L22):** designability is computed against one folding oracle; whether
  ESMFold/AF2/AF3 *disagreement* is amplified specifically on few-step / distilled samples is
  unmeasured.
- **Length ceiling (L24):** headline speedups are single-length (~300); Scaffold-Lab documents
  designability degrades with length, so min-NFE-for-target-designability likely grows with L.
- **Self-conditioning breakdown (L31):** self-conditioning (feeding the previous x̂₀ estimate)
  is poor early in sampling; its interaction with aggressive step reduction is uncharacterized
  for protein structure diffusion (the paper cited to kill this, FastDiSS arXiv:2604.05551, is a
  diffusion *language model* paper — wrong domain).

### 1.3 Scope
- **Covered:** a single audit that measures all five degradation modes on released, open-weight
  few-step / distilled generators across an NFE sweep, and reports each with the statistical
  discipline (multi-length, multi-oracle, multi-seed, bootstrap CIs) the field omits.
- **Non-goals:** proposing a new fast sampler or distillation method (the three method-threads
  are closed); training any model from scratch; a fix for any failure mode (a fix, e.g. the L18
  perceptual loss, is a separate downstream program gated on this audit finding the mode real).
- **Nature of the contribution:** an evaluation / negative-results / benchmark paper — it
  reframes the speedup wave as the object of study rather than competing with it.

## 2. Behavior

The audit takes released generator checkpoints and produces, per failure mode, a curve or
table over the NFE sweep with confidence intervals. It runs entirely at inference time.

- **Inputs:** one or more open-weight generators with a documented multi-step sampler
  (candidates: Proteina, Genie2, FrameFlow/FoldFlow-2/ReQFlow for backbones; the many-step
  setting of each serves as its own reference). NFE grid {2, 4, 8, 50}; lengths {100, 200, 300,
  (600/800 where the checkpoint supports it)}; fixed seed sets per cell.
- **Shared substrate:** for every generated backbone, one designability computation
  (ProteinMPNN 8 seqs @ T=0.1 → ESMFold → Cα scRMSD < 2 Å). All five diagnostics read from this
  same sample set, so the marginal cost of each additional diagnostic is small.
- **Outputs, one per failure mode:**
  - **L18** — Spearman ρ between a frozen-encoder feature distance and per-sample designability,
    vs the same for coordinate distance; positive result = feature distance tracks designability
    better (margin ≥ 0.15).
  - **L27** — precision/recall in the frozen-encoder feature space as NFE drops; collapse =
    recall falls while a Fréchet scalar stays flat.
  - **L22** — per-sample designability label agreement (Cohen's κ) across ESMFold / AF2 / AF3,
    computed separately for few-step vs many-step samples; effect = κ drops faster for few-step.
  - **L24** — designability as a surface over (length × NFE); effect = min-NFE for target
    designability grows with length, shrinking the single-length headline speedup.
  - **L31** — designability gap with self-conditioning on vs off, as a function of NFE; effect =
    the gap widens as NFE drops (interaction), rather than being NFE-independent.
- **Statistical contract (applied uniformly):** every reported number carries a 95% bootstrap
  CI; rankings between settings are reported as CI-overlap, not point estimates. This is the
  L25 discipline (the field's single-seed point estimates hide overlapping CIs).

## 3. Architecture

```
   released generator checkpoint(s)                        BUILT + TDD-verified
   sampled over NFE × length × seed                        this session:
            │ backbones (PDB)                               src/l18/analysis.py
            ▼                                               (spearman_abs,
   ┌──────────────────────────────┐                         bootstrap_ci, adjudicate)
   │ shared designability substrate│                              │
   │ MPNN×8 @T=0.1 → ESMFold →     │                              │ consumes
   │ scRMSD<2Å  (per-sample label) │                              ▼
   └───────────────┬──────────────┘                    ┌────────────────────┐
                   │ (label + backbones reused by all)   │  analysis / stats  │
   ┌───────┬───────┼────────┬────────────┬─────────┐     │  Spearman · P/R ·  │
   ▼       ▼       ▼        ▼            ▼         ▼      │  κ · surface fit · │
 L18     L27     L22      L24          L31      (shared) │  bootstrap CIs     │
 feat-   prec/   cross-   length×NFE   self-cond          └────────┬───────────┘
 dist ρ  recall  oracle κ surface      on/off gap                  ▼
                                                          per-mode curves + CIs
                                                          → the paper's figures
```

- **Generator sampler** *(GPU, not yet built)* — thin wrapper emitting backbones per
  (length × NFE × seed) from a released checkpoint's own multi-step sampler.
- **Designability substrate** *(GPU, not yet built)* — ProteinMPNN → ESMFold → scRMSD; the
  per-sample designable label every diagnostic depends on.
- **Feature encoder** *(partly CPU, not yet built)* — a frozen structure encoder (Foldseek 3Di
  cheapest; ESM-IF; ProteinMPNN encoder) producing the feature vectors L18 and L27 read.
- **Oracle bank** *(GPU + external, not yet built)* — ESMFold plus AF2 and AF3 for the L22
  cross-oracle labels; AF3 access is the one external dependency.
- **Analysis core** *(built, verified — `src/l18/analysis.py`, 8 passing tests)* — absolute
  Spearman, percentile bootstrap CIs, and the pre-registered adjudication rule; extends to the
  P/R, κ, and surface computations. Hardware-independent; already runs locally.

## 4. Data & Interfaces
- **Pre-registration:** `docs/L18_PROTOCOL.md` holds the locked L18 protocol; the bundle needs
  the analogous locked protocol for L27/L22/L24/L31 (thresholds + PASS/KILL per mode) before any
  run, same discipline.
- **Ledger:** `docs/RESEARCH_LEDGER.md` records the five revivals, the two invalid-citation
  kills, and this bundle as candidate **L32**.
- **Analysis contract (existing):** `adjudicate(rho_feat, rho_rmsd) -> Verdict`;
  `spearman_abs(distance, designable) -> float`;
  `bootstrap_ci(distance, designable, n_boot, seed, alpha) -> (lo, hi)`.
- **Headline deliverable:** five figures/tables (one per mode) on shared samples, each with CIs,
  plus the cross-mode observation of whether the modes co-occur (do the same NFE settings that
  collapse coverage also break self-conditioning?).
- **Environment:** `.venv` (numpy 2.5.1, scipy 1.18.0, pytest 9.1.1) for the analysis core;
  generator sampling + ESMFold + AF2/AF3 need a GPU + torch, absent on the current machine.

## 5. Open Questions
- **Is the bundle framing itself scooped?** Individual modes are verified open; a *unified*
  few-step degradation audit has not been checked for prior art (a combined benchmark could
  exist as a mid-2026 workshop paper).
- **Ambition level.** As an evaluation/negative-results paper the natural venue is a workshop or
  a conference eval/benchmark track, not a method-paper slot. Whether that ambition level is
  acceptable is unresolved.
- **AF3 access for L22.** The cross-oracle diagnostic needs AF3 (or AF2 alone if AF3 is
  unavailable); which oracles are reachable is not yet established.
- **Minimum viable subset.** Whether all five modes are run together or a subset (e.g. the three
  purely inference-only ones — L27, L22, L24 — first, deferring L18's encoder choice and L31's
  self-conditioning ablation) is undecided.
- **Compute access.** How the 1×A100 is reached remains unresolved (no `scripts/queue/` in this
  repo); all GPU-side components wait on it.
