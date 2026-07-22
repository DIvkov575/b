# L39 — Phage/Viral Protein ESM-2 Fine-Tune

**Status: VALID, STATISTICALLY CONFIRMED RESULT.** Refined 2026-07-22 with
bootstrap significance testing, a real-benchmark comparison, and a
multi-seed robustness check — see "Refinement" section below for what
changed. Original run: 2026-07-20.

## Hypothesis

ESM-2's UniRef pretraining corpus underrepresents viral/phage protein
sequences (documented in Sawhney et al. 2025, PeerJ, DOI
10.7717/peerj.19919 — "viral proteins... dark matter" in PLM training
data). A cheap masked-LM fine-tune on phage-specific sequences should
measurably close (or invert) a real perplexity gap between phage and
general-protein sequences, using nothing beyond a stock `transformers`
model — no exotic dependencies, no external repo installs.

This directly avoided a dependency trap hit earlier in the same research
thread (ESMFold requiring `openfold`, a GitHub-only install with no PyPI
package, for an unrelated indel-folding idea that was dropped).

## Data

- **Phage:** 20,000 Caudoviricetes (bacteriophage) protein sequences,
  length 50–500 aa, pulled directly from UniProt REST
  (`rest.uniprot.org/uniprotkb/search?query=taxonomy_id:2731619...`,
  cursor-paginated, no auth). Cleaned to 19,980 (standard-residue-only,
  20–512 aa) via `src/l38/phage_data.py`. Split 90/10 → 17,982 train /
  1,998 held-out eval (seed 0, fixed across all training-seed variants).
- **General comparison set:** 5,007 reviewed UniProt sequences excluding
  viral taxonomy (`reviewed:true AND length:[50 TO 500] NOT
  taxonomy_id:10239`), cleaned to 4,990, with a held-out eval slice of
  1,996 — never touched by training, used purely to confirm the model
  isn't just improving everywhere (generic overfitting) but specifically
  on the targeted phage distribution.

## Model / training

- `facebook/esm2_t12_35M_UR50D` (35M params), loaded via plain
  `transformers.AutoModelForMaskedLM` — no LoRA/PEFT, no custom install.
- Two masking-corruption variants compared (see "Refinement" for the
  resolved verdict on which is better — spoiler: neither, robustly):
  - **v1** (`train_phage_esm.py`): always-replace-with-`[MASK]`.
  - **v2** (`train_phage_esm_v2.py`): real BERT-style 80/10/10
    (Devlin et al. 2019) — 80% → `[MASK]`, 10% → random amino acid, 10% →
    left unchanged (still scored in the loss).
- 3 epochs, batch size 16, LR 2e-5, AdamW, full fine-tune (all params).
- **Data-split seed (fixed=0)** vs. **training seed** (`--train-seed`,
  varied 0/1/2 for the robustness check) are separate knobs — see
  `set_train_seed()` in `train_phage_esm.py`, added during refinement so
  masking-pattern/DataLoader-shuffle randomness could be isolated from data
  composition when checking seed sensitivity.
- Eval metric: pseudo-perplexity on held-out sequences using a **fixed,
  seeded mask** per eval call, so baseline and post-training numbers are
  directly comparable (not re-randomized each time).
- Hardware: 1× A10G (g5.xlarge, `i-0659e54e8adc759d3`), ~20 minutes
  wall-clock per training run.

## Perplexity result (seed 0, original run)

| | Baseline | After fine-tune | Δ |
|---|---|---|---|
| Phage pseudo-perplexity (n=1998, held-out) | 12.73 | **5.52** | **−56.6%** |
| General-protein perplexity (n=1996, held-out, untouched) | 10.16 | 11.00 | +8.3% (expected: no longer the training distribution) |

Baseline confirms the hypothesis on real held-out data before any training:
phage sequences are measurably harder for stock ESM-2 than general
proteins. After fine-tuning, phage perplexity drops sharply and **crosses
below** general-protein perplexity.

Full metrics: `src/l38/phage_finetune_out_results.json`,
`src/l38/phage_finetune_v2_out_results.json`.

## Real downstream eval: virion-protein classification

Perplexity alone isn't a publishable result, so this was followed up with
a held-out classification benchmark analogous to ESM-PVP (Li & Liang,
bioRxiv 10.1101/2023.12.29.573676 — see "Refinement" for exactly how this
compares to their real numbers): classify whether a phage protein is a
**virion/structural protein** (capsid, tail, baseplate, etc.) from sequence
alone. Built directly from UniProt's own `KW-0946` (Virion) keyword
annotation — no external repo needed:

- **Data:** 2,994 virion-positive + 2,996 virion-negative Caudoviricetes
  protein sequences pulled via UniProt REST, 5,990 total after dedup,
  80/20 train/test split (stratified, seed 0).
- **Method:** frozen encoder (base ESM-2, or a fine-tuned checkpoint) →
  mean-pooled last-hidden-state embedding → a plain logistic-regression
  probe. No fine-tuning of the probe's backbone; this isolates whether the
  fine-tune *itself* produced better representations.
- Implementation: `src/l38/virion_eval.py`.

### Result (n=5990, held-out test n=1198), with bootstrap significance

| | Accuracy | F1 | AUC |
|---|---|---|---|
| **Base ESM-2 (35M)** | 95.91% | 95.86% | 98.97% |
| **Fine-tuned v1** (naive `[MASK]`-only) | **97.41%** | **97.40%** | **99.53%** |
| **Fine-tuned v2** (BERT-style 80/10/10) | 96.99% | 96.98% | 99.51% |

**Both fine-tuned checkpoints beat the base model, and the gap is
statistically real, not split-luck** (paired bootstrap over the same
held-out test indices, 10,000 resamples, `src/l38/virion_eval.py`'s
`paired_bootstrap_metric_diff` — see Refinement §1 for why paired, not
independent-CI, is the correct test here):

| Comparison | Accuracy Δ (95% CI) | F1 Δ (95% CI) | AUC Δ (95% CI) | Significant? |
|---|---|---|---|---|
| v1 − base | +1.50pp [+0.58, +2.42] | +1.54pp [+0.61, +2.50] | +0.57pp [+0.32, +0.84] | **Yes, all 3 metrics** |
| v2 − base | +1.00pp [0.00, +2.00] | +1.03pp [+0.07, +2.02] | +0.53pp [+0.30, +0.79] | F1/AUC yes; accuracy borderline (CI touches 0) |

Full numbers: `src/l38/virion_eval_results_paired.json`.

## Refinement (2026-07-22) — three follow-ups, all resolved with real data

### 1. Bootstrap confidence intervals (resolved: gap is real)

The original result reported point estimates only, from a single
train/test split — no way to tell if the ~1.5pp accuracy gap was a real
effect or an artifact of which 1,198 sequences landed in that particular
test split. Two things were checked, in order:

- **Independent bootstrap CIs** per model (`bootstrap_metric_ci`,
  10,000 resamples) — these came back *overlapping* (base [94.7%, 97.0%]
  vs. v1 [96.5%, 98.2%]), which looked inconclusive at first.
- **But base/v1/v2 are evaluated on the identical held-out test indices**
  (same seed, same label array length → `sklearn.train_test_split` returns
  identical indices regardless of which model produced the embeddings).
  Comparing independent CIs throws away that pairing and is a strictly
  weaker test. Re-ran as a **paired bootstrap on the per-example prediction
  difference** (`paired_bootstrap_metric_diff`) — this is the statistically
  correct test for "does model B beat model A on the same examples," and it
  cleanly confirmed the gap: v1's improvement clears zero on all 3 metrics,
  v2's clears zero on F1/AUC (accuracy CI lower bound lands exactly at 0.0,
  i.e. borderline).
- Tests: `tests/l38/test_virion_eval.py` (8 tests) — including a synthetic
  case specifically constructed to show the paired test detects a small,
  consistent per-example improvement that an independent-CI comparison
  would likely miss.

### 2. Real ESM-PVP benchmark comparison (resolved: documented, not literally reproduced)

Checked whether L39's virion-classification result could be compared
directly against ESM-PVP's actual published numbers, rather than only
against a self-built comparison task. Found and read the real paper
directly (Li & Liang, bioRxiv 10.1101/2023.12.29.573676, "ESM-PVP:
Identification and classification of phage virion proteins with a large
pretrained protein language model and an MLP neural network"):

- Their binary task (PVP vs. non-PVP) reports **F1 0.9774, AUC 0.99159**
  on a PhaVIP-derived, RefSeq-sourced, time-split test set (train:
  pre-Dec-2020, test: post-Dec-2020) — a different, larger, differently-
  constructed dataset than L39's UniProt-`KW-0946`-based task.
- Their method: ESM-2 **650M** (18.6x larger than L39's 35M), last 4 layers
  fine-tuned, summed (not mean-pooled) embeddings, feeding an MLP head
  (1280→320→80→20→2) — not a frozen-embedding linear probe.
- **The authors' own code/data repo (`github.com/li-bw18/ESM-PVP`, cited
  directly in the paper's text) returns 404 as of this check** — confirmed
  via both the GitHub API and raw githubusercontent fetch on multiple
  branch names. Their exact test set is not currently retrievable, so a
  literal reproduction on their data is not possible right now.
- **Honest comparison, not a literal one:** L39 (97.4% accuracy, AUC 0.995,
  35M frozen model + logistic regression) is in the same performance range
  as ESM-PVP (F1 0.977, AUC 0.992, 650M partially-fine-tuned model + MLP)
  on a *different but task-analogous* benchmark. This is suggestive that
  L39's small-model result is competitive with a much larger, task-specific
  published system — but it is not a head-to-head number and shouldn't be
  cited as beating ESM-PVP on their own benchmark, since that benchmark
  wasn't actually run against.

### 3. v1-vs-v2 masking question (resolved: the gap is seed noise, not real)

The original doc flagged v1 beating v2 by ~0.4pp as unresolved — "may
simply be within noise." Added `--train-seed` (seeds torch/numpy/random for
masking-pattern and DataLoader-shuffle randomness, while keeping the
data-split seed fixed at 0 so all seed variants train/eval on identical
data) via `set_train_seed()`, then trained v1 and v2 at 2 additional seeds
(1, 2) and re-ran the virion eval for each:

| Seed | v1 accuracy | v2 accuracy | Paired diff (v2−v1), 95% CI | Significant? |
|---|---|---|---|---|
| 0 (original) | 97.41% | 96.99% | −0.42pp | not tested directly (only vs. base) |
| 1 | 97.33% | 97.33% | 0.00pp [−0.50, +0.50] | No |
| 2 | 97.33% | 96.99% | −0.33pp [−0.83, +0.17] | No |

**Verdict: no seed shows a statistically significant v1-vs-v2 gap** — the
original "v1 beats v2" observation was exactly the seed-noise artifact the
doc had already flagged as a live possibility. At this model/data scale,
naive always-`[MASK]` corruption and proper BERT-style 80/10/10 corruption
produce statistically indistinguishable downstream classification quality.
The theoretically-motivated v2 fix (avoid over-reliance on the literal
`[MASK]` token, which a frozen-embedding probe never sees) doesn't measurably
help *or* hurt here — a genuine null result on that specific sub-question,
not evidence either recipe is better.

Full multi-seed data: `src/l38/multiseed_v1_vs_v2_results.json`. New tests:
seeding determinism verified directly (`set_train_seed` produces identical
`torch.rand()` draws given the same seed, differs across seeds).

## Honest caveats (remaining, after refinement)

- **ESM-PVP comparison is not head-to-head** (different dataset, different
  model scale, different classifier head) — see Refinement §2. A true
  reproduction would need the authors' dataset, which isn't currently
  retrievable from their cited repo.
- **Novelty vs. prior art:** Sawhney et al. 2025 and PhageContraMLM already
  demonstrate viral/phage-specific PLM fine-tuning improves embeddings —
  this reproduces that general finding on a fresh model/dataset/benchmark
  combination (virion classification specifically) rather than
  establishing something wholly new.
- **No hyperparameter tuning** was done for any training run (3 epochs,
  LR 2e-5, batch 16 throughout, across all seeds) — a tuned configuration
  could plausibly do better in either direction.
- **Perplexity gains and classification gains are separately measured but
  point the same direction** — reassuring, but the classification numbers
  (now bootstrap-confirmed) are the ones that matter for "does this
  actually help."

## Status: shipped, statistically confirmed, benchmark-contextualized

The core result — fine-tuning ESM-2 on 18K UniProt-sourced phage sequences
for ~20 minutes on one A10G produces measurably, significantly better
frozen embeddings for a downstream phage-protein classification task —
now has: (1) bootstrap-confirmed statistical significance via the correct
paired test, (2) an honest, sourced comparison against the real published
ESM-PVP numbers (with the gap in comparability stated plainly, not glossed
over), and (3) a resolved, multi-seed-verified finding that the masking-
recipe choice (v1 vs v2) doesn't matter at this scale. All work completed
end-to-end on the EC2 A10G (`i-0659e54e8adc759d3`) using only plain
`transformers` + `scikit-learn` and direct UniProt REST downloads — no
external repo installs, no exotic dependencies. Checkpoints (v1/v2 ×
3 seeds each) and all result JSONs are saved under `src/l38/`.
