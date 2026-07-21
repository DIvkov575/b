# L40 — Boltz-MSA-Augmentation Ablation

Started as a straight port of [PFold](https://github.com/DIvkov575/PFold)
(commit `f44eecc`)'s data-import/processing layer, then extended into a
controlled ablation: does training PFold's ProteinBERT MLM on Boltz's
MSA-derived homolog sequences (multiple aligned sequences per RCSB structure)
beat training on exactly one raw RCSB sequence per structure? Pre-registered
protocol and results: `docs/L40_PROTOCOL.md`.

**Headline pilot finding (2344 real structures, single seed):** the naive
comparison (Boltz `sequences_per_file=5` vs. baseline `1`) showed boltz
+0.069 val_accuracy — but a steps-matched disambiguation run
(`sequences_per_file=1` for both) collapsed that to −0.004. The effect is
**training volume** (sampling multiple MSA homologs multiplies the training
set), not an intrinsic curation-quality difference in Boltz's `.npz`
pipeline. Full writeup in `docs/L40_PROTOCOL.md`.

## Files

**Ported from PFold (data layer only):**
- `vocab.py` — amino-acid ↔ integer vocabulary (22 AAs + PAD/GAP/MASK/X),
  plus `sequence_to_ids` for tokenizing raw FASTA strings (added for the
  baseline arm — PFold itself never needed this since it only ever read
  from pre-tokenized `.npz` files).
- `msa_data.py` — `MSADataset` (lazy `.npz` loading + BERT-style MLM
  masking), `load_protein_data`, `create_diverse_splits` (file-disjoint
  train/val/test split), `create_dataloaders`. From PFold's `data.py`.
- `model.py` — `ProteinBERT` + `create_model`, from PFold's `model.py`.
- `upload_to_s3.py` — resumable parallel uploader for local `.npz` shards.
  From PFold's `scripts/upload_to_s3.py`, unchanged.
- `download_boltz_msa.sh` — downloads Boltz's public pre-processed RCSB MSA
  `.npz` shards (~250GB total, `s3://boltz1.s3.us-east-2.amazonaws.com`).
  Extracted from PFold's `scripts/instance_setup.sh` into a standalone step.

**Built for the ablation (no PFold equivalent):**
- `mlm_common.py` — BERT-style masking + pad/truncate, shared by both
  dataset classes so masking logic can't drift between the two arms.
- `fetch_rcsb_structures.py` — fetches one canonical single-chain sequence
  per structure from RCSB's FASTA API, for the same structures present in a
  Boltz `.npz` directory. This is the "no MSA augmentation" data source.
- `baseline_data.py` — `RCSBBaselineDataset` + `create_baseline_dataloaders`,
  mirroring `msa_data.py`'s interface exactly so the same training loop
  consumes either data source unmodified.
- `train_pilot.py` — shared training script, `--variant {boltz,baseline}`,
  identical hyperparameters/seed for both arms.
- `compare_results.py` — final-epoch delta between two `train_pilot.py` runs.

## Expected data format

Each Boltz `.npz` shard has:
- `sequences`: structured array with `res_start`/`res_end` fields indexing
  into `residues`.
- `residues`: structured array with a `res_type` field (integer, per
  `vocab.AA_VOCAB`).

The RCSB baseline is a JSONL file, one line per structure:
`{"structure_id": "8u3n_a", "pdb_id": "8U3N", "chain_id": "A", "sequence": "MGSS..."}`.

## Reproducing the pilot

Exact commands and provenance (commit SHAs, environment, raw metrics) are
pinned in `docs/L40_PROTOCOL.md` — this is the abbreviated version:

```bash
# 1. Get a slice of Boltz's data (full tar is ~107GB; HTTP Range works for a subset)
mkdir -p data/l40_pilot/npz
curl -s -r 0-1600000000 "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar" \
  | tar -x --strip-components=1 -C data/l40_pilot/npz
# Drop the last file in the listing — the byte range cuts it off mid-file,
# so it's a truncated/invalid .npz (numpy will raise BadZipFile on load).

# 2. Fetch the matching RCSB baseline sequences (2344 real requests, ~0.1s apart —
# takes a few minutes; resumable, safe to re-run if interrupted)
.venv-l38/bin/python -m src.l40.fetch_rcsb_structures \
  --npz-dir data/l40_pilot/npz --out data/l40_pilot/rcsb_baseline.jsonl \
  --max-structures 2344 --sleep-seconds 0.05

# 3a. Train the boltz variant (main comparison: MSA-homolog augmented)
.venv-l38/bin/python -m src.l40.train_pilot --variant boltz \
  --data-path data/l40_pilot/npz --max-files 2344 --epochs 3 --batch-size 32 \
  --sequences-per-file 5 --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 --seed 0

# 3b. Train the baseline variant (1 sequence/structure, no augmentation)
.venv-l38/bin/python -m src.l40.train_pilot --variant baseline \
  --data-path data/l40_pilot/rcsb_baseline.jsonl --max-files 2344 --epochs 3 --batch-size 32 \
  --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 --seed 0

# 3c. Disambiguation run: boltz with sequences_per_file=1 — isolates curation
# from training-volume by matching the baseline's step count exactly
.venv-l38/bin/python -m src.l40.train_pilot --variant boltz \
  --data-path data/l40_pilot/npz --max-files 2344 --epochs 3 --batch-size 32 \
  --sequences-per-file 1 --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 --seed 0 \
  --out-path src/l40/pilot_out/boltz_seqs1_results.json

# 4. Compare (repeat with --boltz pointed at boltz_seqs1_results.json for the
# disambiguation comparison)
.venv-l38/bin/python -m src.l40.compare_results \
  --boltz src/l40/pilot_out/boltz_results.json \
  --baseline src/l40/pilot_out/baseline_results.json
```

Requires a torch-enabled venv (`.venv-l38`, Python 3.11, `torch==2.13.0`). On
Apple Silicon / MPS, `train_pilot.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1`
itself — `nn.TransformerEncoder`'s padding-mask fast path calls an op MPS
doesn't implement (`aten::_nested_tensor_from_mask_left_aligned`).

Tests: `tests/l40/` (`.venv-l38/bin/python -m pytest tests/l40/ -v` — 49
tests, ~85s, no network calls, no GPU required though MPS/CUDA is used if
present).
