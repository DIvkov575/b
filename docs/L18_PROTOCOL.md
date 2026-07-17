# L18 — Structural-LPIPS for Protein Backbone Distillation

**Pre-registered Phase 2b MVP kill-gate.** Locked 2026-07-12, *before* any run.
Do not edit thresholds after seeing data — that is the L25 failure mode. If the
protocol is wrong, note the flaw, kill/pass by the stated rule, and re-register a v2.

Scoop status (2026-07-12): **NOT SCOOPED** (see RESEARCH_LEDGER Phase-2a-bis note).
Nearest neighbor to differentiate against: **Protein FID (arXiv:2505.08041)** —
distributional, eval-only, validated vs CATH/Foldseek/OT, *not* designability, *not*
per-sample, *not* a loss. L18's defensible slice = per-sample perceptual distance +
**designability correlation** + **use as a distillation loss**.

---

## The claim under test (this MVP tests only the first half)

**H1 (diagnosis).** A per-sample distance in a *frozen protein-structure encoder's*
feature space (the "protein-LPIPS" analog) tracks designability of few-step / distilled
backbones *better than* coordinate/frame-space distances (Kabsch-RMSD, geodesic-L2).

The training-loss half (H2, "fix collapse") is **out of scope for the MVP** — it only
gets built if H1 passes. This keeps us from over-building before falsification.

---

## Inputs (all fixed before run)

- **Teacher:** one released open-weight backbone generator with a public checkpoint
  and a documented multi-step sampler. Candidates, in preference order:
  1. Proteina (if weights + sampler are actually runnable on 1×A100)
  2. Genie2
  3. FrameFlow / FoldFlow-2 / ReQFlow
  Pick the FIRST one that loads and samples end-to-end in mock-free mode. Record which.
- **Lengths:** L ∈ {100, 200, 300}. (Single length is the L24 mistake; 3 is the cheap hedge.)
- **NFE grid:** {2, 4, 8, 50}. 50 = "many-step reference"; the reference output is the
  distillation target the few-step samples are compared against, per length, per seed.
- **N samples:** 100 backbones per (length × NFE) cell → 3 × 4 × 100 = **1200 backbones**.
  Reference cells (NFE=50) reuse the same 100 seeds so few-step vs reference is paired.
- **Sampling seeds:** fixed list `[0..99]` per cell. Reused across NFE so pairing holds.

## Designability protocol (LOCKED — this is the L25-critical part)

For each generated backbone:
1. **Sequence design:** ProteinMPNN, **8 sequences**, **temperature 0.1**, default model
   `v_48_020`, no fixed positions, no bias.
2. **Folding oracle:** ESMFold (primary). Fold all 8 sequences.
3. **scRMSD:** Cα-RMSD (Kabsch) between each ESMFold prediction and the generated backbone.
4. **Per-backbone designable** iff `min over 8 seqs of scRMSD < 2.0 Å`.
5. **Designability of a cell** = fraction of its 100 backbones that are designable.
6. **Cross-oracle spot-check (L22 hedge, cheap):** re-fold the NFE=8, L=300 cell with
   AF2 (single-seq or ColabFold) and report κ agreement with ESMFold labels. Reported,
   not gating.

## Distances computed per few-step backbone (vs its paired NFE=50 reference)

- **d_RMSD:** Kabsch Cα-RMSD to the reference backbone.
- **d_geo:** geodesic-L2 in frame space (log-map on SE(3) frames), for completeness.
- **d_feat:** the candidate perceptual distance = L2 in a **frozen structure encoder's**
  feature space. Encoder candidates (compute all that load; pick the best a priori by
  cheapest-to-run, break ties by designability correlation on a 100-sample dev slice
  held out from the reported set):
  - Foldseek 3Di token embeddings (cheapest, no GPU for inference)
  - ESM-IF (inverse-folding) encoder embeddings
  - ProteinMPNN encoder embeddings
  Record which encoder is used for the headline number and report all as a table.

## Correlation analysis

For each distance d ∈ {d_RMSD, d_geo, d_feat}, compute **Spearman ρ between d and
per-backbone designability** (binary; use point-biserial-equivalent Spearman) pooled
across all NFE<50 cells, and also per-NFE. Bootstrap 95% CI (10k resamples over backbones).

## PASS / KILL rule (pre-registered — do not move)

- **KILL-A (nothing to add):** if `|ρ(d_RMSD, designability)| > 0.6` pooled — coordinate
  distance already predicts designability well, no room for a perceptual distance. Dead.
- **KILL-B (no signal):** if `|ρ(d_feat, designability)| < 0.3` pooled — the feature
  distance doesn't track designability. Dead.
- **PASS:** iff `|ρ(d_feat, designability)| > 0.3` AND
  `|ρ(d_feat, ·)| − |ρ(d_RMSD, ·)| ≥ 0.15` (feature distance beats RMSD by a margin).
- **Ambiguous band** (feat > 0.3 but margin in [0, 0.15)): report honestly, treat as a
  WEAK pass — proceed to H2 only if the margin is positive at every individual NFE.

## Cost / feasibility

- No training in this MVP. Pure inference + folding.
- 1200 backbones × 8 MPNN seqs = 9600 sequences to fold with ESMFold. On 1×A100 this is
  hours, not days. Foldseek/ESM-IF feature extraction is cheap.
- Whole gate: **~1 day of A100 wall-clock** once the teacher checkpoint loads.

## What this machine can vs cannot do

- **This laptop:** no GPU, no torch. Can build/lint the harness, run mock-mode, write the
  analysis (Spearman + bootstrap) against synthetic fixtures.
- **A100:** required for teacher sampling + ESMFold. The harness ships mock-mode so the
  full control flow + analysis is verified locally before the GPU run.

## Deliverable regardless of outcome

A **NFE × designability × d_feat** table + the three Spearman ρ's with CIs. Even a KILL is
a citable negative result ("coordinate distance suffices" / "no perceptual signal"), which
is more than the field currently reports (L20/L25 gaps).
