# HLD: Structural-LPIPS for Protein Backbone Distillation (`l18-paper`)

**Date:** 2026-07-12 · **Author:** divkov · **Status:** Draft

## 1. Overview

### 1.1 Background
- A wave of Oct-2025 → mid-2026 papers made protein backbone generators *fast*: few-step
  distillation and manifold-native flow maps (SiD-Protein arXiv:2510.03095, Generalised
  Flow Maps arXiv:2510.21608, Riemannian MeanFlow arXiv:2602.07744, Riemannian
  Consistency Model arXiv:2510.00983). These deliver 10–20× speedups by cutting the number
  of function evaluations (NFE) from ~50 to a handful.
- Every one of them measures quality with **designability** (generate → ProteinMPNN
  sequences → fold with ESMFold → self-consistency RMSD < 2 Å) and trains/distills with a
  **coordinate- or frame-space distance** to the teacher: L2, Kabsch-RMSD, geodesic-L2 on
  SE(3) frames.
- In image generation, the analogous distances (pixel L2) are known to be poor proxies for
  perceptual quality; the field moved to **LPIPS** — a distance in a *frozen encoder's*
  feature space that correlates with human perception and is used both to evaluate and as a
  training loss. **No protein-structure analog of LPIPS exists.**
- Verified this session (adversarial scoop-recheck, direct abstract fetch): the closest
  prior art is **Protein FID (arXiv:2505.08041)** — a Fréchet distance in a learned
  structure latent space. It is *distributional* (a set-level FID, not a per-sample
  distance), *evaluation-only* (explicitly not a training loss), and validated against
  CATH/Foldseek/OT clusters — **not against designability**.

### 1.2 Problem Statement
- Few-step / distilled backbone generators are suspected to buy designability by collapsing
  toward mean / canonical structures, but the losses used to build and judge them cannot
  *see* that collapse: a coordinate distance treats a collapsed-but-close backbone as good.
- There is no per-sample, designability-correlated distance for protein structures — the
  quantity that would (a) diagnose which few-step samples are quietly degraded and (b) serve
  as a training signal that penalizes collapse where coordinate losses don't.
- The open, defensible slice (what Protein FID leaves untaken):
  - **per-sample** perceptual distance, not a set-level FID;
  - validated to **correlate with designability**, not with fold-family clusters;
  - usable as a **distillation training loss**.

### 1.3 Scope
- **Covered:** the paper's thesis, the two hypotheses it rests on (H1 diagnosis, H2 fix),
  the pre-registered MVP that gates the whole effort, the metrics, and the compute path.
- **This document describes the research program and its paper**, not an implementation.
- **Non-goals:** a new fast sampler (the speedups already exist — this is a distance/loss
  that plugs into them); an all-atom or sequence-space method; beating SOTA designability
  numbers (the contribution is the distance and what it reveals, not a leaderboard entry).

## 2. What the paper claims and how it is tested

**Thesis.** A distance in a frozen protein-structure encoder's feature space — the protein
analog of LPIPS — is a designability-correlated, per-sample quality signal for backbone
generators that coordinate/frame distances miss, and it can both diagnose and reduce
mean-structure collapse in few-step / distilled models.

The thesis decomposes into two hypotheses, tested in sequence with a hard gate between them.

- **H1 — Diagnosis (the MVP; pre-registered in `docs/L18_PROTOCOL.md`).**
  On one released teacher, generate backbones across NFE ∈ {2, 4, 8, 50} and lengths
  L ∈ {100, 200, 300}, 100 samples per cell. For each backbone measure designability and
  three distances to the paired 50-step reference: Kabsch-RMSD (`d_rmsd`), frame geodesic-L2
  (`d_geo`), and the candidate feature distance (`d_feat`). Test whether `d_feat` tracks
  designability better than the coordinate distances.
  - **PASS** iff |ρ(`d_feat`, designable)| > 0.3 **and** it beats |ρ(`d_rmsd`, ·)| by ≥ 0.15.
  - **KILL-A** if |ρ(`d_rmsd`, ·)| > 0.6 (coordinate distance already suffices — nothing to add).
  - **KILL-B** if |ρ(`d_feat`, ·)| < 0.3 (no signal).
  - Cost: ~1 day on 1×A100, **no training**.

- **H2 — Fix (only if H1 passes).**
  Add `d_feat` as an auxiliary distillation loss on a few-step student and test whether it
  recovers designability and diversity lost to collapse, versus the same student trained
  with coordinate loss alone. Requires short fine-tuning / LoRA on the A100.

**Outcome is publishable either way.** A KILL is a citable negative result — "coordinate
distance suffices" or "no perceptual signal in protein structure encoders" — reported with
CIs the field currently omits. A PASS is the positive result: the first protein-structure
perceptual distance, validated against designability.

## 3. Architecture (of the experiment that produces the paper)

```
                    ┌─────────────────────────────────────────────┐
                    │  released teacher backbone generator (A100)  │
                    │  sampled at NFE ∈ {2,4,8,50}, L ∈ {100,..}   │
                    └───────────────┬─────────────────────────────┘
                                    │ backbones (PDB)
              ┌─────────────────────┼──────────────────────────┐
              ▼                     ▼                          ▼
     ┌────────────────┐   ┌──────────────────┐      ┌───────────────────┐
     │ designability  │   │  coord distances │      │ feature distance  │
     │ MPNN×8 @T=0.1  │   │  d_rmsd, d_geo   │      │ d_feat = L2 in    │
     │ → ESMFold      │   │  (to NFE=50 ref) │      │ frozen encoder    │
     │ → scRMSD<2Å    │   └────────┬─────────┘      │ (Foldseek/ESM-IF/ │
     └───────┬────────┘            │                │  ProteinMPNN enc) │
             │  designable (0/1)   │  distances     └─────────┬─────────┘
             └──────────────┬──────┴──────────────────────────┘
                            ▼
              ┌──────────────────────────────────┐
              │  analysis core (runs anywhere)    │   ← BUILT + TDD-verified
              │  src/l18/analysis.py              │     this session
              │  spearman_abs · bootstrap_ci ·    │
              │  adjudicate → PASS/WEAK/KILL-A/-B │
              └──────────────────────────────────┘
```

- **Teacher sampler** *(GPU, not yet built)* — thin wrapper around one released open-weight
  generator; the first that loads and samples end-to-end (preference order Proteina → Genie2
  → FrameFlow/FoldFlow-2/ReQFlow). Emits backbones as PDB per (length × NFE × seed).
- **Designability harness** *(GPU, not yet built)* — ProteinMPNN (8 seqs, T=0.1, `v_48_020`)
  → ESMFold → Cα Kabsch scRMSD; per-backbone designable = min-over-8 scRMSD < 2 Å.
  Cross-oracle spot-check with AF2 on one cell (reported, not gating).
- **Distance module** *(partly GPU, not yet built)* — `d_rmsd` and `d_geo` are pure geometry;
  `d_feat` runs each backbone through a **frozen** structure encoder and takes L2 in feature
  space. Encoder candidates: Foldseek 3Di (CPU-cheap), ESM-IF, ProteinMPNN encoder.
- **Analysis core** *(built, verified — `src/l18/analysis.py`, 8 passing tests)* — consumes
  per-backbone (distance, designable) arrays, computes absolute Spearman ρ, percentile
  bootstrap CIs, and applies the pre-registered rule to emit the verdict. Hardware-independent.

## 4. Data & Interfaces

- **Pre-registration:** `docs/L18_PROTOCOL.md` — the locked H1 protocol (inputs, seeds,
  designability definition, thresholds). Frozen before any run; edits require a re-registered v2.
- **Ledger:** `docs/RESEARCH_LEDGER.md` — L18 status `picked`; records the 2026-07-12
  scoop-recheck verdict and Protein FID as the differentiation target.
- **Analysis contract:** `adjudicate(rho_feat, rho_rmsd) -> Verdict{status, rho_feat,
  rho_rmsd, margin, reason}`; `spearman_abs(distance, designable) -> float`;
  `bootstrap_ci(distance, designable, n_boot, seed, alpha) -> (lo, hi)`.
- **Headline deliverable (paper's core table):** NFE × designability × `d_feat`, with the
  three Spearman ρ's and their 95% CIs, per length and pooled.
- **Environment:** `.venv` (numpy 2.5.1, scipy 1.18.0, pytest 9.1.1) for the analysis core.
  Teacher sampling + ESMFold need a GPU + torch, absent on the current machine.

## 5. Open Questions
- **Compute access.** How the 1×A100 is reached is unresolved — no `scripts/queue/` in this
  repo and the ssh config was not inspected. The GPU-side components cannot run until the
  access path is known.
- **Teacher choice.** Which released generator actually loads and samples end-to-end on the
  available A100 (weights + runnable multi-step sampler) is unverified; the protocol fixes a
  preference order but the first-that-works is determined at run time.
- **Encoder choice for `d_feat`.** Which frozen encoder (Foldseek 3Di / ESM-IF / ProteinMPNN)
  carries the headline number is decided a priori on a held-out dev slice; not yet run.
- **Residual scoop risk.** Scoop-recheck confidence is medium-high; a mid-2026
  OpenReview/workshop appendix adding a Protein-FID-style loss was not fully ruled out.
