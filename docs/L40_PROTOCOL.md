# L40 — Boltz-MSA-Augmentation Ablation Pilot

**Pre-registered 2026-07-21, before the real run.** Locks the comparison
method and pinned-split invariant before seeing pilot results.

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
