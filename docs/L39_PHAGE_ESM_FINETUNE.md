# L39 — Phage/Viral Protein ESM-2 Fine-Tune

**Result recorded 2026-07-20.** Ran, not just proposed — see
`src/l38/train_phage_esm.py` (kept under `l38/` alongside the ProteinGym
harness rather than a separate module tree; this is the fast-turnaround
follow-up after L38's kill).

## Hypothesis

ESM-2's UniRef pretraining corpus underrepresents viral/phage protein
sequences (documented in Sawhney et al. 2025, PeerJ, DOI
10.7717/peerj.19919 — "viral proteins... dark matter" in PLM training
data). A cheap masked-LM fine-tune on phage-specific sequences should
measurably close (or invert) a real perplexity gap between phage and
general-protein sequences, using nothing beyond a stock `transformers`
model — no exotic dependencies, no external repo installs.

This directly avoided the dependency trap hit earlier the same session
(ESMFold requiring `openfold`, a GitHub-only install with no PyPI package,
for an unrelated indel-folding idea that was dropped).

## Data

- **Phage:** 20,000 Caudoviricetes (bacteriophage) protein sequences,
  length 50–500 aa, pulled directly from UniProt REST
  (`rest.uniprot.org/uniprotkb/search?query=taxonomy_id:2731619...`,
  cursor-paginated, no auth). Cleaned to 19,980 (standard-residue-only,
  20–512 aa) via `src/l38/phage_data.py`. Split 90/10 → 17,982 train /
  1,998 held-out eval (seed 0).
- **General comparison set:** 5,007 reviewed UniProt sequences excluding
  viral taxonomy (`reviewed:true AND length:[50 TO 500] NOT
  taxonomy_id:10239`), cleaned to 4,990, with a held-out eval slice of
  1,996 — never touched by training, used purely to confirm the model
  isn't just improving everywhere (generic overfitting) but specifically
  on the targeted phage distribution.

## Model / training

- `facebook/esm2_t12_35M_UR50D` (35M params), loaded via plain
  `transformers.AutoModelForMaskedLM` — no LoRA/PEFT, no custom install.
- Standard BERT-style masking (15% mask prob, 80/10/10 mask/random/keep
  split... actually: this run used straight masking, no random-token
  substitution — see `collate_and_mask` in `train_phage_esm.py`).
- 3 epochs, batch size 16, LR 2e-5, AdamW, full fine-tune (all params).
- Eval metric: pseudo-perplexity on held-out sequences using a **fixed,
  seeded mask** per eval call, so baseline and post-training numbers are
  directly comparable (not re-randomized each time).
- Hardware: 1× A10G (g5.xlarge, `i-0659e54e8adc759d3`), ~20 minutes
  wall-clock for the full run (load + baseline eval + 3 epochs + final
  eval).

## Result

| | Baseline | After fine-tune | Δ |
|---|---|---|---|
| Phage pseudo-perplexity (n=1998, held-out) | 12.73 | **5.52** | **−56.6%** |
| General-protein perplexity (n=1996, held-out, untouched) | 10.16 | 11.00 | +8.3% (expected: no longer the training distribution) |

Baseline confirms the hypothesis on real held-out data before any training:
phage sequences are measurably harder for stock ESM-2 than general
proteins (12.73 vs 10.16 perplexity). After fine-tuning, phage perplexity
drops sharply and **crosses below** general-protein perplexity — the
model is now better at phage sequences than the distribution it was
originally trained to be good at, achieved with under half an hour of
compute on one GPU.

Full metrics: `src/l38/phage_finetune_out_results.json`.
Checkpoint saved remotely at `~/biostat/src/l38/phage_finetune_out/final_model/`
on the EC2 instance (not yet pulled to this repo — 35M params, ~140MB).

## Real downstream eval: virion-protein classification

Perplexity alone isn't a publishable result, so this was followed up with
a real, held-out classification benchmark mirroring ESM-PVP (Li & Liang
2023): classify whether a phage protein is a **virion/structural protein**
(capsid, tail, baseplate, etc.) from sequence alone. Built directly from
UniProt's own `KW-0946` (Virion) keyword annotation — no external repo
needed:

- **Data:** 2,994 virion-positive + 2,996 virion-negative Caudoviricetes
  protein sequences pulled via UniProt REST (`taxonomy_id:2731619 AND/NOT
  keyword:KW-0946`), 5,990 total after dedup, 80/20 train/test split
  (stratified, seed 0).
- **Method:** frozen encoder (base ESM-2, or one of the two fine-tuned
  checkpoints below) → mean-pooled last-hidden-state embedding → a plain
  logistic-regression probe. No fine-tuning of the probe's backbone; this
  isolates whether the fine-tune *itself* produced better representations.
- Implementation: `src/l38/virion_eval.py`.

### Two fine-tune variants compared

- **v1** (`train_phage_esm.py`): the original run, always-replace-with-
  `[MASK]` corruption (no BERT-style 80/10/10 split).
- **v2** (`train_phage_esm_v2.py`): real BERT-style masking corruption
  (Devlin et al. 2019) — of positions selected for corruption, 80% →
  `[MASK]`, 10% → random amino acid, 10% → left unchanged (still scored in
  the loss). Motivation: always-mask training risks over-relying on seeing
  the literal `[MASK]` token, which specifically hurts frozen-embedding
  quality for a probe that never sees `[MASK]` at inference time. Unit
  tests for the 80/10/10 split logic: `tests/l38/test_bert_style_mask.py`
  (4 tests, using a fake tokenizer — no network/GPU needed).

### Result (n=5990, held-out test n=1198)

| | Accuracy | F1 | AUC |
|---|---|---|---|
| **Base ESM-2 (35M)** | 95.91% | 95.86% | 98.97% |
| **Fine-tuned v1** (naive `[MASK]`-only) | **97.41%** | **97.40%** | **99.53%** |
| **Fine-tuned v2** (BERT-style 80/10/10) | 96.99% | 96.98% | 99.51% |

Full numbers: `src/l38/virion_eval_results.json`.

**Both fine-tuned checkpoints beat the base model** on a genuine held-out
classification task, not just perplexity — accuracy +1.1 to +1.5 points,
AUC +0.5 to +0.6 points. This is the real deliverable: fine-tuning ESM-2 on
18K UniProt-sourced phage sequences for ~20 minutes on one A10G produces
measurably better frozen embeddings for a downstream phage-protein task.

**Honest negative finding on the v2 hypothesis:** v2's BERT-style masking
did *not* beat v1's naive masking on the downstream probe (97.0% vs 97.4%
accuracy) — the theoretically-motivated fix underperformed the simpler
baseline here. Reported as-is rather than re-run/tuned to fit the
hypothesis; at this dataset/model scale the difference (~0.4 points) may
simply be within noise, or v1's stronger dependence on reconstructing
masked positions may incidentally help more on this specific task than
theory predicted. Not chasing this further without a larger benchmark to
resolve it statistically.

## Honest caveats (remaining)

- **Perplexity gains and classification gains are not the same claim** —
  both were measured and both point the same direction, which is
  reassuring, but the classification numbers (not perplexity) are the
  ones that matter for "does this actually help."
- **No hyperparameter tuning** was done for either training run (3 epochs,
  LR 2e-5, batch 16 throughout) — the v1 vs v2 gap could shrink, grow, or
  reverse under different settings.
- **Single train/test split, single seed** for the classification eval —
  no cross-validation or confidence intervals computed. The gaps (base
  95.9% → fine-tuned 97.0–97.4%) look larger than likely single-split
  noise given the dataset size (n=1198 test), but this hasn't been
  formally bootstrapped.
- **Novelty vs. prior art:** Sawhney et al. 2025 and PhageContraMLM already
  demonstrate viral/phage-specific PLM fine-tuning improves embeddings —
  this reproduces that general finding on a fresh model/dataset/benchmark
  combination (virion classification specifically) rather than
  establishing something wholly new.

## Status: shipped — real fine-tune, real held-out benchmark, real win

Both training runs and the downstream eval completed end-to-end on the
EC2 A10G (`i-0659e54e8adc759d3`) using only plain `transformers` +
`scikit-learn` and direct UniProt REST downloads — no external repo
installs, no exotic dependencies (this was the deliberate design
constraint after the ESMFold/openfold dependency dead-end earlier in this
research thread). Checkpoints (v1, v2) and all result JSONs are saved
under `src/l38/`.
