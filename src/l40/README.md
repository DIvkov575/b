# L40 — Boltz-MSA-Augmentation Ablation

Started as a straight port of [PFold](https://github.com/DIvkov575/PFold)
(commit `f44eecc`)'s data-import/processing layer, then extended into a
controlled ablation: does training PFold's ProteinBERT MLM on Boltz's
MSA-derived homolog sequences (multiple aligned sequences per RCSB structure)
beat training on exactly one raw RCSB sequence per structure? Pre-registered
protocol and results: `docs/L40_PROTOCOL.md`.

Two ablations live here, both in `docs/L40_PROTOCOL.md`:

1. **Training-volume ablation (2344 real structures, single seed):** does
   sampling multiple MSA homologs per structure (vs. exactly 1 RCSB
   sequence) help, and is any apparent gain real signal or just more data?
   The naive comparison (Boltz `sequences_per_file=5` vs. baseline `1`)
   showed boltz +0.069 val_accuracy — but a steps-matched disambiguation run
   collapsed that to −0.004, and a total-volume-matched run (baseline run
   15 epochs instead of 3) shrank it further to +0.009. **The effect is
   almost entirely training volume, not data quality** — Boltz's specific
   `.npz` curation adds ~nothing beyond a plain RCSB fetch.
2. **Full architectural ablation (500 real structures, single seed):** given
   that the *content* of Boltz's `.npz` files (taxonomy, deletion
   bookkeeping, multiple aligned homologs) was mostly unused above, this
   second ablation builds a real MSA-consuming architecture (ported from
   Boltz's own model code, Pairformer removed — see below) and tests each
   mechanism independently: profile feature, deletion feature, cross-
   sequence attention, and all combined. **The profile feature (simple
   per-position amino-acid frequency across homologs) gave the largest gain
   (+0.051 val_accuracy, ~67% relative) — larger than the much more
   expensive cross-sequence-attention mechanism (+0.010) or deletion
   features (~0, within noise).**

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

**Built for the architectural ablation (ported from Boltz's real model
source, verified against `jwohlwend/boltz`'s GitHub — not guessed):**
- `msa_features.py` — `compute_deletion_features` (`has_deletion =
  raw_count > 0`, `deletion_value = π/2·arctan(raw_count/3)` — exact
  transform from Boltz's `featurizerv2.py`) and `compute_profile`
  (per-position amino-acid frequency across homologs).
- `taxonomy_sampling.py` — picks a taxonomically diverse homolog subset per
  structure (Boltz uses `taxonomy` only as a selection key, never a model
  input — verified against `trunkv2.construct_paired_msa`).
- `msa_module.py` — `Transition`, `OuterProductMean`, `PairWeightedAveraging`,
  `MSALayer`, `MSAModule`: Boltz's real MSA-embedding stack (exact tensor
  shapes/dims ported from `model/layers/{transition,outer_product_mean,
  pair_averaging}.py` and `model/modules/trunkv2.py`), with the structure-
  specific `PairformerNoSeqLayer` call removed from `MSALayer` — nothing
  downstream needs a refined pair representation for an MLM task. Found and
  fixed a real gap during testing: with `z` at zero and `msa_blocks=1`,
  homolog info couldn't reach the query before the module returned (see
  `docs/L40_PROTOCOL.md` for detail) — fixed by seeding `z` from the raw MSA
  embedding before the first block.
- `msa_model.py` — `MSAAwareProteinBERT`: query embedding + optional profile
  branch + optional `MSAModule`, each independently toggleable, then a
  standard transformer encoder + MLM head.
- `msa_aware_data.py` — `MSAAwareDataset` + `create_msa_aware_dataloaders`:
  per structure, samples homologs (taxonomy-diverse), computes deletion/
  profile features from *all* available homologs (matching Boltz's real
  featurizer, computed before any subsampling), MLM-masks only the query row.
- `train_msa_ablation.py` — trains `MSAAwareProteinBERT` with any
  combination of `--use-deletion-features`/`--use-profile`/`--use-msa-module`.

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

## Reproducing the architectural ablation

```bash
# Same data as above (data/l40_pilot/npz). One call per arm:
.venv-l38/bin/python -m src.l40.train_msa_ablation \
  --data-path data/l40_pilot/npz --max-files 500 --epochs 2 --batch-size 8 \
  --max-length 128 --msa-depth 5 --d-model 128 --n-layers 4 --n-heads 4 --d-ff 512 \
  --msa-s 64 --token-z 32 --msa-blocks 1 --seed 0 \
  --out-path src/l40/pilot_out/msa_ablation_A_baseline.json      # baseline

# ...add --use-profile for arm B, --use-msa-module for arm C,
# --use-msa-module --use-deletion-features for arm D,
# --use-msa-module --use-deletion-features --use-profile for arm E.
```

**Hardware note:** `OuterProductMean`'s pair tensor is `O(N²)` in sequence
length — `max_length=512` (used above) OOMs on a 48GB MPS pool. This
ablation uses `max_length=128` and a smaller structure count (500, not
2344) to fit; see `docs/L40_PROTOCOL.md` for the exact tradeoff.

Requires a torch-enabled venv (`.venv-l38`, Python 3.11, `torch==2.13.0`). On
Apple Silicon / MPS, both training scripts set `PYTORCH_ENABLE_MPS_FALLBACK=1`
themselves — `nn.TransformerEncoder`'s padding-mask fast path calls an op MPS
doesn't implement (`aten::_nested_tensor_from_mask_left_aligned`).

Tests: `tests/l40/` (`.venv-l38/bin/python -m pytest tests/l40/ -v` — 89
tests, ~2-3 min, no network calls, no GPU required though MPS/CUDA is used
if present).
