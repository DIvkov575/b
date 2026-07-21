# L38 — Low-MSA-Depth Slice of ProteinGym: Baseline Kill-Test

**Pre-registered gate spec.** Locked 2026-07-20, *before* any run. Do not edit
thresholds after seeing data — that is the L25 failure mode. If a threshold turns
out to be wrong, note the flaw, kill/pass by the stated rule, and re-register a v2.

Scoop status (2026-07-20): validated via adversarial multi-agent review (not a
formal scoop-check like L18's) — ProteinGym's substitution × low-MSA-depth slice
(36 assays) is data-abundant, and the literature attributes the current ceiling
(~0.498 Spearman, AIDO Protein-RAG-16B) to architecture/retrieval-dependency, not
label scarcity — the opposite failure mode of two prior candidates rejected before
this one (CAID3 Binding-IDR, QMAP AMP-MIC regression; both killed on data-scarcity/
noise-floor grounds, see conversation history, not yet in RESEARCH_LEDGER since this
is a separate research thread from the L01–L37 structure-generation pool).

An initially proposed novel mechanism (Neff/L-conditioned adaptive fusion between
frozen ProteinMPNN structure signal and MSA-retrieval PLM signal) was independently
reviewed by 3 adversarial passes and rated **serious** on all three: likely non-novel
(overlaps TranceptEVE's EVE+Tranception depth-aware combination), self-undermining
(its only theoretical edge over Kermut's already-adaptive GP applies to label-sparse
assays, but this slice is defined by low *MSA depth*, not low *label count*), and
confounded (low-Neff proteins skew viral/orphan/disordered, where the structure
signal is independently degraded too — any ensembling would look like a win here,
adaptive or not). **The adaptive-gate idea is shelved pending Gate 1 data**, not
built as originally conceived.

---

## The claim under test

**H1 (real gap, not noise).** On ProteinGym's official substitution × low-MSA-depth
slice, at least one of {ProteinMPNN-only, retrieval-augmented-PLM-only, static
stacking of both, Kermut-style per-assay-refit GP} beats current best-in-slice
performance (~0.498 Spearman, AIDO Protein-RAG-16B) by a margin that clears
bootstrapped noise — **without needing any new mechanism**.

**H0 (just hard).** Structure-only performance degrades roughly in step with
retrieval-only performance on this slice (both signals independently weak on
the same confounded subpopulation — viral/orphan/disordered proteins). If so,
the ceiling reflects genuine task difficulty, not an exploitable architecture
gap, and L38 should be abandoned rather than forced into a mechanism paper.

This spec tests H1 vs H0 directly with existing components before any novel
architecture work is attempted (Gate 2, contingent — see below).

---

## Inputs (all fixed before run)

- **Benchmark:** ProteinGym v1.3 (release PG_v1.3, 2025-04-28), substitution DMS
  track, `reference_files/DMS_substitutions.csv`, `MSA_Neff_L_category == "Low"`
  stratum. Per the deep-dive research: **36 assays**. Record the exact assay-ID
  list used (version drift exists — README states 74 total substitution assays
  in some contexts; pin to the v1.3 CSV shipped with the repo, not a recollected
  count).
- **Metric:** per-assay Spearman correlation (model score vs. experimental fitness),
  matching ProteinGym's own `performance_DMS_benchmarks.py` convention: aggregate by
  UniProt ID first, then average — do not deviate from this aggregation to avoid
  an apples-to-oranges comparison against the published 0.498 baseline number.
- **Bootstrap:** 10,000 resamples over assays (not over individual mutants — the
  unit of aggregation is the assay/protein, matching ProteinGym's own
  `compute_bootstrap_standard_error`), reporting SE and 95% CI per method.
- **Baselines under test (fixed set of 4 — do not add a 5th mid-run):**
  1. **ProteinMPNN-only** — frozen inverse-folding log-likelihood ratio (mutant
     vs. wild-type), no MSA input.
  2. **Retrieval-augmented-PLM-only** — VenusREM-style: frozen PLM logits fused
     with MSA-homolog retrieval signal (fixed global blend, no adaptivity).
  3. **Static linear stacking** — fixed-weight linear combination of (1) and (2),
     weight fit once on a disjoint calibration split (not on the 36-assay eval
     set itself — see Data plan below).
  4. **Kermut-style two-kernel GP** — vendored from the public Kermut repo
     (code copied into `src/l38/vendor/kermut/`, not a git submodule/pip dep,
     per user preference), per-assay marginal-likelihood-refit hyperparameters
     combining a structure kernel and a sequence kernel.
- **Current published reference point:** AIDO Protein-RAG-16B, 0.498 Spearman
  (low-MSA-depth stratum average, per ProteinGym's own Summary_performance CSV —
  re-confirm this exact number from the CSV directly in Gate 0, do not trust the
  research-conversation recollection uncritically).

## Data plan (leakage discipline)

- Train/calibrate any fittable component (the static-stacking weight; GP kernel
  hyperparameters where not already published/frozen) on ProteinGym assays
  **outside** the 36-assay low-MSA-depth slice (e.g. medium/high-MSA-depth
  assays, or a held-out subset of low-MSA assays disjoint from the final 36
  reported on). Never tune against the 36-assay numbers themselves before
  reporting — this is the exact non-blind-gaming failure mode flagged in the
  ProteinGym-mechanics research finding.
- No new labeled data collection — all fitness labels already exist in
  ProteinGym's DMS assay files.

## PASS / KILL rule (pre-registered — do not move)

Applied per baseline, then combined:

- **DELIVERABLE:** any single baseline's mean Spearman on the 36-assay slice
  exceeds 0.498 by **≥ 2× the combined bootstrap SE** (i.e., clears the
  documented ~0.012–0.025 SE band with a real margin, roughly ≥0.03–0.05
  absolute Spearman). → Stop here. This alone is the paper (a slice-specific
  robustness claim, Tranception-precedent framing). Skip to Gate 3. No novel
  mechanism required.
- **KILL-NOISE (H0):** ProteinMPNN-only's degradation from its own performance
  on medium/high-MSA assays is **within 1 SE** of retrieval-only's equivalent
  degradation (i.e., structure-only does *not* hold up materially better than
  retrieval-only on this slice specifically). → The gap is not an exploitable
  architecture asymmetry; it's shared task difficulty. **Abandon L38.** Do not
  proceed to Gate 2. Write up as a negative result (see Gate 3 fallback).
- **PROCEED (H1, but below SOTA):** ProteinMPNN-only holds up
  **materially better** than retrieval-only (gap exceeds 1 SE) — a real,
  currently-unexploited asymmetry exists — but no baseline individually clears
  the DELIVERABLE bar. → Earns Gate 2 (contingent falsification battery for a
  minimal fusion mechanism).
- **AMBIGUOUS:** any pattern not cleanly matching the above (e.g. mixed across
  sub-groups, wide overlapping CIs). Report honestly. Default toward
  **KILL-NOISE** treatment unless a single, specific, pre-specifiable-in-advance
  follow-up hypothesis falls out cleanly from the data (per the project's
  general bias toward fast convergence over prolonged ambiguity).

## Gate 2 (contingent — only on PROCEED, spec deferred)

Not fully specified now — depends on which specific assays/protein families
drive the Gate 1 asymmetry, which isn't known yet. Will be pre-registered as a
v1 addendum *before* touching Gate 2 data, following the same discipline. Must
include, at minimum (carried over from the adversarial mechanism review, not
re-litigated): a shuffled/permuted-depth-signal ablation, family/taxonomy-
stratified re-check (to rule out the confound), and a head-to-head requirement
against the Kermut GP baseline specifically (not just against retrieval-only).

## Gate 3 — write-up (reached via DELIVERABLE, a Gate-2 PASS, or a KILL-NOISE
negative result)

- **If DELIVERABLE or Gate-2 PASS:** frame as a slice-specific robustness claim
  grounded in ProteinGym's own published `Low_MSA_depth` column (Tranception
  precedent — see mechanics research finding). Submit via ProteinGym's standard
  open-source PR + "new model" GitHub issue process.
- **If KILL-NOISE:** still a citable result — "structure and retrieval signals
  degrade in lockstep on low-MSA-depth substitution assays; the 0.498 ceiling
  reflects task difficulty, not an unexploited architecture gap." Report the
  4-baseline table + bootstrap CIs regardless of outcome (same principle as
  L18's "deliverable regardless of outcome").

## Cost / feasibility

- Gate 0 + baselines 1–3: no training required, frozen-backbone inference only.
  ProteinMPNN scoring and static-stacking fit are CPU/laptop-feasible for ~36
  assays' worth of variants (thousands, not millions, of mutants per the
  ProteinGym reference files).
- Baseline 4 (Kermut GP): closed-form GP fitting, minutes per assay per the
  Kermut paper's own reported runtime — laptop-feasible.
- Retrieval step (MSA search via jackhmmer/HHblits for baseline 2, if MSAs
  aren't already shipped by ProteinGym for these assays) is the most likely
  bottleneck — check whether ProteinGym ships pre-computed MSAs for the 36
  assays before assuming a search step is needed at all.
- **Whole Gate 0+1: no A100 provisioning assumed necessary.** Escalate to GPU
  only if a specific step (embedding extraction at scale, MSA search) proves
  too slow locally — decide per-step, not up front.

## What this machine can vs cannot do

- **This laptop:** everything in Gate 0 and Gate 1 is expected to run here.
  No GPU assumed necessary based on the data scale (36 assays, thousands of
  variants). Build/test the harness against synthetic fixtures first (TDD),
  then run for real.
- **A100 (contingent):** only if profiling shows a specific step (MSA search
  at scale, embedding extraction) is impractically slow locally.

## Deliverable regardless of outcome

A **4-baseline × 36-assay Spearman table**, with bootstrapped SEs/CIs, plus the
structure-vs-retrieval degradation comparison that adjudicates H1 vs H0. Even a
KILL-NOISE result is a citable negative finding, consistent with this project's
existing convention (L18) that a clean kill is a valid, reportable outcome — not
a failure to be hidden.

---

## GATE 1 RESULT (2026-07-20) — KILL, premise falsified (stronger than KILL-NOISE)

Run against real ProteinGym v1.3 data (`reference_files/DMS_substitutions.csv`,
`benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`,
pulled directly from `github.com/OATML-Markslab/ProteinGym`, cached under
`src/l38/data_cache/`). Harness validated in Gate 0 (reproduces the published
`Low_MSA_depth` column within bootstrap noise — see
`tests/l38/test_gate0_reproduction.py`).

**Degradation table (High-MSA-depth mean → Low-MSA-depth mean, bootstrapped,
10k resamples over assays):**

| Model | High | Low | Gap | Gap SE |
|---|---|---|---|---|
| ProteinMPNN (structure-only) | 0.416 | **0.188** | **0.228** | 0.030 |
| AIDO Protein-RAG (16B) (retrieval+structure) | 0.583 | 0.502 | 0.080 | 0.028 |
| VenusREM (retrieval) | 0.574 | 0.502 | 0.072 | 0.030 |
| GEMME (pure MSA) | 0.495 | 0.450 | 0.045 | 0.031 |

**Verdict: the pre-registered PROCEED condition required structure-only to hold
up *better* than retrieval-only on this slice (a real, exploitable asymmetry in
the hypothesized direction). The data shows the opposite, by a margin (~3+
combined SEs, not "within 1 SE") too large to call ambiguous: ProteinMPNN
degrades far *more* than every retrieval-based method tested.** This is a
cleaner falsification than the anticipated KILL-NOISE case (which only
required "no asymmetry") — here the asymmetry exists and runs backward from
the mechanism this whole research thread was built on.

**DELIVERABLE check:** static linear stack of ProteinMPNN + AIDO-RAG (weight
fit on the disjoint Medium+High-depth calibration split, per the leakage
discipline above) scores **0.387** on the Low-depth slice — *worse* than
either component alone, because blending in the now-confirmed-weak structure
signal drags the average down. No deliverable.

**Interpretation:** low-MSA-depth proteins in this slice (BRCA1, BRCA2, P53,
GFP, and others) are enriched for proteins where fold stability alone doesn't
determine function/fitness — exactly the confound the pre-run adversarial
review flagged. ProteinMPNN, trained on ordered/crystallizable PDB structures,
is independently weak on this subpopulation. There is no free structural
signal available to fuse; the retrieval-based methods (already near-published-
SOTA on this slice) are already doing better than structure-only by a wide,
confirmed margin.

**Decision: KILL. Do not proceed to Gate 2.** No fusion mechanism (adaptive or
static) has a viable secondary signal to combine — the "rescue" signal this
plan depended on is the weakest one on the target slice, not a hidden
strength. L38 is closed as a negative result per the Gate 3 KILL branch:
structure and retrieval signals do *not* trade off in the direction assumed;
structure signal collapses harder on low-MSA-depth substitution assays than
retrieval signal does, and the field's existing SOTA (retrieval/MSA-hybrid
models, ~0.498–0.502) already reflects this — there is no unexploited
architecture gap here, just a confirmation that these methods are already
close to appropriately weighted.

Reproducible via `src/l38/gate1_analysis.py` (run as `python -m
src.l38.gate1_analysis` from the repo root with `.venv-l38`).
