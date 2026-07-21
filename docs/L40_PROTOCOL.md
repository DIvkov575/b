# L40 — Boltz-MSA-Augmentation Ablation Pilot

**Pre-registered 2026-07-21, before the real run.** Locks the comparison
method and pinned-split invariant before seeing pilot results.

## Provenance

- **Code:** `src/l40/` in this repo, commits `76d23de0`..`06f5cbff`
  (2026-07-21). Ported from [PFold](https://github.com/DIvkov575/PFold)
  commit `f44eecc`; ablation-specific code (baseline fetcher/dataset,
  shared trainer, comparator) has no PFold equivalent — see `src/l40/README.md`
  for the file-by-file breakdown.
- **Environment:** `.venv-l38` — Python 3.11.15, `torch==2.13.0`, run on
  Apple Silicon (MPS backend, no CUDA available).
- **Tests:** `tests/l40/` — 49 tests, all passing as of the commits above
  (`.venv-l38/bin/python -m pytest tests/l40/ -v`).
- **Raw data:**
  - Boltz slice: 2344 real `.npz` shards, byte-range-fetched from
    `s3://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar`
    (public, no auth). Not committed (large binaries) — `data/l40_pilot/`
    is gitignored; re-fetch via the command in `src/l40/README.md`.
  - RCSB baseline: `data/l40_pilot/rcsb_baseline.jsonl`, fetched live via
    `src/l40/fetch_rcsb_structures.py` against
    `https://www.rcsb.org/fasta/chain/{PDBID}.{CHAIN}` — 2344/2344 succeeded,
    0 failures. Also gitignored; regenerate via the same script.
- **Result artifacts (committed, small JSON):** `src/l40/pilot_out/
  {boltz,baseline,boltz_seqs1}_results.json` — per-epoch train/val
  loss+accuracy for all three training runs referenced below.
- **Reproduction commands:** `src/l40/README.md` § "Reproducing the pilot"
  has the exact CLI invocations, including the disambiguation run.

## Hypothesis

Training PFold's ProteinBERT MLM on Boltz's MSA-derived homolog sequences
(multiple aligned sequences per RCSB structure) improves held-out MLM
accuracy/loss over training on exactly one raw RCSB sequence per structure —
holding architecture, hyperparameters, structure population, and file-level
train/val/test split fixed.

## Method

- **Model A (boltz):** `src/l40/msa_data.py` pipeline, `sequences_per_file=5`
  (samples up to 5 MSA-hit sequences per training structure).
- **Model B (baseline):** `src/l40/baseline_data.py` pipeline, exactly 1
  sequence per structure (the canonical RCSB chain sequence, fetched via
  `src/l40/fetch_rcsb_structures.py`).
- Same structure population for both, capped at `--max-files N` — the first
  N `.npz` filenames (sorted) in the Boltz data directory.
- Same file-level train/val/test split for both (pinned via
  `create_diverse_splits`'s `random.seed(42)`, verified identical in
  `tests/l40/test_split_pinning.py`).
- Same architecture (`d_model`, `n_layers`, `n_heads`, `d_ff`), optimizer,
  learning rate, batch size, epoch count, mask probability.
- Same `torch.manual_seed` before model construction for both variants
  (`train_pilot.py`'s `--seed`, default 0) — both models start from
  identical initial weights, so a metric difference is attributable to the
  data, not to which variant happened to draw a luckier init.
- Metric: held-out (val) MLM accuracy and loss, final epoch.

## Pilot scale

A few thousand structures (`--max-files`), a handful of epochs, run locally
on MPS (no CUDA available on this machine; `PYTORCH_ENABLE_MPS_FALLBACK=1` is
required — `nn.TransformerEncoder`'s padding-mask fast path calls an op MPS
doesn't implement, `aten::_nested_tensor_from_mask_left_aligned`; the fallback
runs that one op on CPU instead of crashing). This is a first-pass signal
check, not a scaled/definitive result.

## What counts as what

- **Boltz helps:** final val_accuracy(boltz) − val_accuracy(baseline) is
  positive and larger than run-to-run noise (assessed qualitatively at pilot
  scale — no repeated-seed variance estimate yet; a real effect claim would
  need that before writeup).
- **No effect / baseline wins:** delta ≈ 0 or negative — the extra MSA
  homologs don't help this model/data-scale combination, at least not
  without more compute.
- Either outcome is a legitimate, reportable pilot result. This is a first
  cheap pass to decide if scaling up is worth it, not a publication-grade
  claim in itself.

## Known limitations of the pilot

- Single seed per variant — no variance estimate.
- Small `max_files`/epoch count relative to PFold's own full-scale training
  (`MAX_FILES=151040`, 250 epochs) — a pilot null result doesn't rule out an
  effect that only appears at scale.
- RCSB fetch rate-limited client-side (`--sleep-seconds`) to avoid hammering
  their API — fetching baseline data for thousands of structures takes
  real wall-clock time (minutes, not seconds).
- **Gradient-steps confound (discovered post-run, see Result below):** at
  `sequences_per_file=5`, the boltz variant trains on 5x more sequences per
  epoch than the baseline (8200 vs 1640, same 1640 structures). Matching
  epoch count therefore does not match total gradient steps — the result
  conflates "MSA homology diversity helps" with "more total training data/
  steps helps." A cleaner isolation would either cap boltz at
  `sequences_per_file=1` (pure curation-only comparison, expected near-zero
  per the original design discussion) or match total training steps between
  variants directly.

## Result (2026-07-21)

Ran on the full pilot slice: 2344 real Boltz-processed RCSB structures
(HTTP range-fetched from Boltz's public S3 tar), 2344 matching RCSB baseline
sequences (fetched live via `fetch_rcsb_structures.py`, 0 failures). Pinned
split confirmed identical on real data: 1640 train / 351 val / 353 test
structures for both variants (`tests/l40/test_split_pinning.py`-equivalent
check re-run directly against the real data before training).

Hyperparameters (identical for both): `d_model=128, n_layers=4, n_heads=4,
d_ff=512, batch_size=32, lr=1e-4, epochs=3, mask_prob=0.15, seed=0`.
Boltz variant used `sequences_per_file=5`.

| | boltz (MSA-augmented) | baseline (RCSB, 1 seq/structure) |
|---|---|---|
| train sequences/epoch | 8200 | 1640 |
| final train_loss | 2.7241 | 2.9456 |
| final val_loss | 2.6933 | 2.8723 |
| final val_accuracy | 0.1981 | 0.1291 |

**val_accuracy_delta = +0.0690 (boltz higher, ~53% relative), val_loss_delta
= −0.1790 (boltz lower/better).** Both metrics moved consistently in favor
of the boltz variant across all 3 epochs, not just the final one — boltz's
val_accuracy was already ahead by epoch 1 (0.170 vs 0.098) and the gap held.

**Verdict per the pre-registered criteria: "Boltz helps"** — a positive
val_accuracy delta of this size, consistent across epochs, on a real
(non-synthetic) 2344-structure slice.

**But this does not yet isolate MSA-augmentation as the cause**, per the
gradient-steps confound above: boltz also saw 5x more training sequences
per epoch. This pilot answers "does Boltz's pipeline as typically configured
(`sequences_per_file=5`) beat the no-augmentation baseline" — yes, clearly —
but not yet "is that because of homology diversity specifically, or because
of more training volume." Disambiguation run below.

## Disambiguation run (2026-07-21): curation-only, steps-matched

Reran the boltz variant with `sequences_per_file=1` — same 2344 structures,
same pinned split, same hyperparameters/seed, but now exactly 1 sequence per
structure (1640 train sequences/epoch, matching the baseline's step count
exactly). This isolates Boltz's specific extraction/curation from its
homology-augmentation effect.

| | boltz (seqs_per_file=1) | baseline (RCSB, 1 seq/structure) |
|---|---|---|
| train sequences/epoch | 1640 | 1640 |
| final train_loss | 2.9414 | 2.9456 |
| final val_loss | 2.8765 | 2.8723 |
| final val_accuracy | 0.1249 | 0.1291 |

**val_accuracy_delta = −0.0042, val_loss_delta = +0.0042** — both
indistinguishable from zero at this pilot's scale (single seed, no variance
estimate, but the magnitude here is ~16x smaller than the sequences_per_file=5
delta and has the opposite sign on accuracy).

## Combined conclusion

The full-pipeline advantage found in the first run (val_accuracy +0.0690) is
attributable almost entirely to **training-volume** (5x more sequences per
epoch from sampling multiple MSA homologs), not to any intrinsic curation-
quality difference between Boltz's `.npz` pipeline and a plain RCSB fetch —
confirming the original design-phase prediction for the "curation-only" cut
(expected near-zero effect) and refuting a naive read of the first run as
evidence that Boltz's processing itself is higher-quality data.

**Practical implication:** if PFold's goal is masked-LM pretraining quality,
the effect being tested here isn't "does Boltz's data pipeline matter" —
it's "does sampling multiple MSA-homolog sequences per structure act as a
cheap data-augmentation multiplier," which this pilot answers **yes** (at
pilot scale, single seed). That's a real and useful finding, but it's a
different claim than "Boltz processing improves per-structure signal
quality," which this pilot found **no evidence for**.

**Caveats before treating either finding as final:** single seed per run (no
variance estimate — a repeat with a different seed could shift these deltas
non-trivially at this small a pilot scale), 3 epochs only, and the
5x-augmentation win could plausibly not hold at PFold's full training scale
(150K structures, 250 epochs) where the baseline would also see much more
data in absolute terms.

## Volume-accounting run (2026-07-21): baseline at matched total training instances

The `sequences_per_file=1` disambiguation above matched *epoch count* and
*structures*, but under-trained the baseline in absolute terms: boltz-5 saw
8200 seqs/epoch × 3 epochs = 24,600 total training instances, while the
`sequences_per_file=1` cut only gave baseline 1640 × 3 = 4,920 — a 5x
smaller total training budget, not a matched one. This run closes that gap
directly: baseline trained for 15 epochs on its native 1640 seqs/epoch
(1640 × 15 = 24,600 — the *same* total instance count as boltz-5's 3-epoch
run), just repeating each of its 1640 single sequences 15 times instead of
seeing 5 distinct MSA homologs 3 times each.

| | boltz (5 seqs/file × 3 epochs) | baseline (1 seq/file × 15 epochs) |
|---|---|---|
| total train instances | 24,600 | 24,600 |
| distinct sequences/structure | 5 | 1 |
| final train_loss | 2.7241 | 2.7178 |
| final val_loss | 2.6933 | 2.6956 |
| final val_accuracy | 0.1981 | 0.1891 |

**val_accuracy_delta = +0.0090, val_loss_delta = −0.0024** — the delta
shrinks by ~87% from the naive +0.0690 to +0.0090 once total training
volume is matched, and val_loss is now a coin-flip either way. The
baseline's own epoch-by-epoch curve (epochs 1–15: val_accuracy 0.098 →
0.191, plateauing from epoch ~8 onward in the 0.185–0.191 band) shows it
converging to nearly the same place as boltz-5 reached in fewer, more
varied epochs — repetition of one sequence gets most of the way there,
just needs more passes to do it.

**Updated combined conclusion:** across all three cuts (naive, curation-
only/steps-matched-per-epoch, and volume-matched/total-instances-matched),
the picture is consistent: Boltz's specific `.npz` extraction and curation
contributes ~nothing beyond a plain RCSB fetch (the curation-only cut,
−0.004). What sampling 5 MSA homologs per structure buys is mostly
**more total gradient steps early**, reaching the baseline's eventual
plateau faster (3 epochs vs. ~8) rather than reaching meaningfully higher —
the residual +0.009 at matched total volume is small enough that a
single-seed pilot can't distinguish it from noise. If speed-of-convergence
(reaching a given accuracy in fewer wall-clock epochs) matters for the
downstream use case, MSA augmentation is a legitimate, real lever; if only
the eventual ceiling matters, this pilot found no evidence it raises it.
