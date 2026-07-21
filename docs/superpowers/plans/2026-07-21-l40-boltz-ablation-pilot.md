# L40 — Boltz-MSA-Augmentation Ablation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train two PFold-architecture PLMs that are identical except for one variable — whether pretraining data includes Boltz's MSA-derived homolog sequences per structure — and measure the effect on held-out MLM accuracy/loss, at pilot scale (a few thousand structures, single GPU/MPS, a few hours).

**Architecture:** Reuse the already-ported `src/l40/msa_data.py` (Boltz `.npz` loader) unchanged for Model A. Add a parallel baseline path: fetch one canonical RCSB sequence per structure (same structure population, same file-level train/val/test split, pinned via a shared deterministic split function) via RCSB's FASTA API, then a matching `RCSBBaselineDataset` with `sequences_per_file=1`. Port PFold's `ProteinBERT` model. One shared training script trains both variants with identical hyperparameters and writes per-epoch metrics to JSON. A comparison script reports the final delta.

**Tech Stack:** Python, PyTorch (`.venv-l38` — has `torch==2.13.0`, `numpy`, `requests`, `pytest`), MPS device (no local CUDA), RCSB REST API (`https://www.rcsb.org/fasta/chain/{PDBID}.{CHAIN}`), Boltz public S3 tar via HTTP Range requests (no AWS credentials needed, listing is blocked but ranged GET works).

**Validity invariant (do not violate):** Model A and Model B must see the *same* set of structures, split into train/val/test by the *same* assignment. This is enforced by building both datasets' `file_sequences` dict key order from the identical sorted list of Boltz `.npz` filenames (capped to the same `max_files`), then feeding both through `create_diverse_splits` (already in `msa_data.py`, seeded `random.seed(42)`), and reusing existing determinism tests as the guardrail. Model A samples up to `max_seqs_per_file_train` MSA-homolog sequences per training file; Model B always has exactly 1 sequence per file (that's what "no augmentation" means) — the *file-level* split is what must match, not the per-file sequence count.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/l40/mlm_common.py` | Shared BERT-style masking + pad/truncate functions (extracted so both dataset classes use identical masking logic — DRY, and correctness-critical since a masking-logic mismatch would confound the ablation). |
| `src/l40/msa_data.py` | *(existing, modified)* `MSADataset` delegates to `mlm_common` instead of its own copies. Behavior-preserving refactor — existing 14 tests must still pass unmodified. |
| `src/l40/vocab.py` | *(existing, modified)* Add `sequence_to_ids(seq: str) -> np.ndarray`, mapping one-letter AA codes to `AA_VOCAB` ints (unknown letters → `X`). |
| `src/l40/fetch_rcsb_structures.py` | Standalone script: given a directory of Boltz `.npz` files, fetch the single canonical sequence per structure from RCSB, write a resumable JSONL. Pure/testable helpers (`parse_structure_id`, `parse_fasta_response`) separated from the network-calling loop. |
| `src/l40/baseline_data.py` | `load_baseline_protein_data` (mirrors `load_protein_data`'s dict shape/order), `RCSBBaselineDataset` (mirrors `MSADataset`'s `__getitem__` contract), `create_baseline_dataloaders`. |
| `src/l40/model.py` | Ported `ProteinBERT` + `create_model` from PFold's `model.py`, unchanged. |
| `src/l40/train_pilot.py` | One script, `--variant {boltz,baseline}`, trains `ProteinBERT` with identical hyperparameters on either data source, logs per-epoch train/val loss + val accuracy to `src/l40/pilot_out/{variant}_results.json`. |
| `src/l40/compare_results.py` | Loads both JSON result files, computes the final-epoch delta, prints a table. |
| `docs/L40_PROTOCOL.md` | Pre-registered spec: hypothesis, pinned-split invariant, what "Boltz helps" vs "no effect" looks like — written before the real run. |
| `tests/l40/test_mlm_common.py` | New. |
| `tests/l40/test_msa_data.py` | *(existing)* must still pass after the refactor. |
| `tests/l40/test_vocab.py` | New. |
| `tests/l40/test_fetch_rcsb_structures.py` | New — mocks network, no real HTTP calls. |
| `tests/l40/test_baseline_data.py` | New. |
| `tests/l40/test_model.py` | New — smoke test on tiny dims. |
| `tests/l40/test_train_pilot.py` | New — full pipeline on tiny synthetic data, 1 epoch, both variants, real (small) training step. |

---

## Task 1: Extract shared MLM masking/padding into `mlm_common.py`

**Files:**
- Create: `src/l40/mlm_common.py`
- Test: `tests/l40/test_mlm_common.py`
- Modify: `src/l40/msa_data.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/l40/test_mlm_common.py
import numpy as np

from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate


class TestApplyMlmMasking:
    def test_masked_prob_zero_leaves_sequence_unchanged_and_no_labels(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        rng = np.random.RandomState(0)
        masked, labels = apply_mlm_masking(sequence, mask_prob=0.0, mask_token=23, rng=rng)
        assert np.array_equal(masked, sequence)
        assert (labels == -100).all()

    def test_masked_prob_one_labels_every_position(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        rng = np.random.RandomState(0)
        masked, labels = apply_mlm_masking(sequence, mask_prob=1.0, mask_token=23, rng=rng)
        assert (labels != -100).all()
        assert np.array_equal(labels, sequence)

    def test_deterministic_given_same_rng_state(self):
        sequence = np.arange(1, 21, dtype=np.int32)
        masked1, labels1 = apply_mlm_masking(sequence, 0.5, 23, np.random.RandomState(42))
        masked2, labels2 = apply_mlm_masking(sequence, 0.5, 23, np.random.RandomState(42))
        assert np.array_equal(masked1, masked2)
        assert np.array_equal(labels1, labels2)


class TestPadOrTruncate:
    def test_pads_short_sequence_with_pad_value(self):
        sequence = np.array([1, 2, 3], dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=6, pad_value=0)
        assert np.array_equal(result, [1, 2, 3, 0, 0, 0])

    def test_truncates_long_sequence(self):
        sequence = np.arange(10, dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=4, pad_value=0)
        assert np.array_equal(result, [0, 1, 2, 3])

    def test_exact_length_is_unchanged(self):
        sequence = np.array([5, 6, 7], dtype=np.int32)
        result = pad_or_truncate(sequence, max_length=3, pad_value=0)
        assert np.array_equal(result, sequence)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_mlm_common.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.mlm_common'`

- [ ] **Step 3: Write the implementation**

```python
# src/l40/mlm_common.py
"""BERT-style MLM masking and pad/truncate logic shared by every dataset in l40.

Extracted from PFold's data.py (MSADataset._apply_mlm_masking / _pad_sequence) so
both the Boltz-MSA dataset and the RCSB-baseline dataset apply identical masking —
a masking-logic mismatch between the two would confound the ablation.
"""
from typing import Tuple

import numpy as np


def apply_mlm_masking(sequence: np.ndarray, mask_prob: float, mask_token: int,
                       rng) -> Tuple[np.ndarray, np.ndarray]:
    """rng must expose .random() -> float and .randint(low, high) -> int
    (np.random.RandomState and the np.random module both satisfy this)."""
    masked_sequence = sequence.copy()
    labels = np.full_like(sequence, -100, dtype=np.int32)

    for i in range(len(sequence)):
        if rng.random() < mask_prob:
            labels[i] = sequence[i]

            # BERT-style corruption: 80% mask token, 10% random AA, 10% unchanged.
            rand_val = rng.random()
            if rand_val < 0.8:
                masked_sequence[i] = mask_token
            elif rand_val < 0.9:
                masked_sequence[i] = rng.randint(1, 22)

    return masked_sequence, labels


def pad_or_truncate(sequence: np.ndarray, max_length: int, pad_value: int) -> np.ndarray:
    if len(sequence) > max_length:
        return sequence[:max_length]
    if len(sequence) < max_length:
        padding = np.full(max_length - len(sequence), pad_value, dtype=sequence.dtype)
        return np.concatenate([sequence, padding])
    return sequence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_mlm_common.py -v`
Expected: 6 passed

- [ ] **Step 5: Refactor `MSADataset` to delegate to the shared functions**

In `src/l40/msa_data.py`, replace the body of `_apply_mlm_masking` and `_pad_sequence`:

```python
from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate
```

```python
    def _apply_mlm_masking(self, sequence: np.ndarray, seq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.fixed_seed is not None:
            rng = np.random.RandomState(self.fixed_seed + seq_idx)
        else:
            rng = np.random
        return apply_mlm_masking(sequence, self.mask_prob, self.mask_token, rng)

    def _pad_sequence(self, sequence: np.ndarray, is_labels: bool = False) -> np.ndarray:
        pad_value = -100 if is_labels else self.pad_token
        return pad_or_truncate(sequence, self.max_length, pad_value)
```

Remove the now-unused `random` import from `msa_data.py` if nothing else in the file uses it (check first — `create_diverse_splits` uses the stdlib `random` module, a different import than `np.random`; keep that one).

- [ ] **Step 6: Run the full existing l40 suite to confirm no regression**

Run: `.venv-l38/bin/python -m pytest tests/l40/ -v`
Expected: all previously-passing tests (14 from `test_msa_data.py`) + 6 new ones = 20 passed

- [ ] **Step 7: Commit**

```bash
git add src/l40/mlm_common.py src/l40/msa_data.py tests/l40/test_mlm_common.py
git commit -m "refactor(l40): extract shared MLM masking/padding into mlm_common.py"
```

---

## Task 2: Add `sequence_to_ids` to `vocab.py`

**Files:**
- Modify: `src/l40/vocab.py`
- Test: `tests/l40/test_vocab.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/l40/test_vocab.py
import numpy as np

from src.l40.vocab import AA_VOCAB, sequence_to_ids


def test_sequence_to_ids_maps_standard_residues():
    result = sequence_to_ids("ARN")
    assert list(result) == [AA_VOCAB['A'], AA_VOCAB['R'], AA_VOCAB['N']]


def test_sequence_to_ids_returns_int32_array():
    result = sequence_to_ids("ARN")
    assert result.dtype == np.int32


def test_sequence_to_ids_maps_unknown_letters_to_X():
    result = sequence_to_ids("AUZ")  # U, Z are non-standard/ambiguous codes
    assert list(result) == [AA_VOCAB['A'], AA_VOCAB['X'], AA_VOCAB['X']]


def test_sequence_to_ids_is_case_insensitive():
    assert list(sequence_to_ids("arn")) == list(sequence_to_ids("ARN"))


def test_sequence_to_ids_empty_string_returns_empty_array():
    result = sequence_to_ids("")
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_vocab.py -v`
Expected: FAIL with `ImportError: cannot import name 'sequence_to_ids'`

- [ ] **Step 3: Implement**

Append to `src/l40/vocab.py`:

```python
import numpy as np

_LETTER_TO_ID = {k: v for k, v in AA_VOCAB.items() if len(k) == 1}
_UNKNOWN_ID = AA_VOCAB['X']


def sequence_to_ids(seq: str) -> np.ndarray:
    """Maps a one-letter amino-acid sequence to AA_VOCAB integer ids.
    Unknown/non-standard letters (e.g. U, Z, B, O, J) map to 'X'."""
    return np.array(
        [_LETTER_TO_ID.get(c.upper(), _UNKNOWN_ID) for c in seq],
        dtype=np.int32,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_vocab.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/l40/vocab.py tests/l40/test_vocab.py
git commit -m "feat(l40): add sequence_to_ids for tokenizing raw FASTA sequences"
```

---

## Task 3: `ProteinBERT` model port

**Files:**
- Create: `src/l40/model.py`
- Test: `tests/l40/test_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/l40/test_model.py
import torch

from src.l40.model import ProteinBERT, create_model


def test_forward_returns_logits_of_expected_shape():
    model = ProteinBERT(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    input_ids = torch.randint(1, 24, (3, 20))
    attention_mask = torch.ones(3, 20)

    out = model(input_ids, attention_mask)

    assert out['logits'].shape == (3, 20, 24)
    assert out['loss'] is None


def test_forward_with_labels_computes_finite_loss():
    model = ProteinBERT(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    input_ids = torch.randint(1, 24, (3, 20))
    attention_mask = torch.ones(3, 20)
    labels = torch.full((3, 20), -100, dtype=torch.long)
    labels[:, 0] = 5

    out = model(input_ids, attention_mask, labels)

    assert torch.isfinite(out['loss'])


def test_create_model_returns_proteinbert_instance():
    model = create_model(vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=20)
    assert isinstance(model, ProteinBERT)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.model'`

- [ ] **Step 3: Implement** (straight port of PFold's `model.py`, no changes)

```python
# src/l40/model.py
"""ProteinBERT MLM model. Ported unchanged from PFold (github.com/DIvkov575/PFold,
commit f44eecc) model.py."""
import math

import torch
import torch.nn as nn


class ProteinBERT(nn.Module):
    def __init__(self, vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, max_length: int = 512, dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pe)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlm_head = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)

        if attention_mask is not None:
            attention_mask = (attention_mask == 0)

        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.norm(x)

        logits = self.mlm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.vocab_size), labels.view(-1))

        return {'loss': loss, 'logits': logits, 'hidden_states': x}


def create_model(vocab_size: int = 24, d_model: int = 256, n_layers: int = 6,
                  n_heads: int = 8, d_ff: int = 1024, max_length: int = 512,
                  dropout: float = 0.1) -> ProteinBERT:
    model = ProteinBERT(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, d_ff=d_ff, max_length=max_length, dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model w/ {total_params:,} total parameters")
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_model.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/l40/model.py tests/l40/test_model.py
git commit -m "feat(l40): port ProteinBERT model from PFold"
```

---

## Task 4: RCSB structure-sequence fetcher

**Files:**
- Create: `src/l40/fetch_rcsb_structures.py`
- Test: `tests/l40/test_fetch_rcsb_structures.py`

**Context:** Boltz `.npz` filenames follow the pattern `{pdbid}_{chain}.npz` (lowercase), e.g. `8u3n_a.npz`. RCSB's per-chain FASTA endpoint is `https://www.rcsb.org/fasta/chain/{PDBID}.{CHAIN}` (uppercase PDB ID, uppercase chain letter), returning a FASTA block:
```
>8U3N.A|Cytochrome P450-SU1|Micromonospora sp. MW-13 (2094022)
MGSSHHHHHHSS...
```
(Confirmed live in the design/exploration phase — this is real, not assumed.)

- [ ] **Step 1: Write the failing tests**

```python
# tests/l40/test_fetch_rcsb_structures.py
import json
from unittest.mock import MagicMock

import pytest

from src.l40.fetch_rcsb_structures import (
    fetch_one,
    load_existing_ids,
    parse_fasta_response,
    parse_structure_id,
)


class TestParseStructureId:
    def test_splits_pdbid_and_chain_from_npz_filename(self):
        pdb_id, chain_id = parse_structure_id("8u3n_a.npz")
        assert pdb_id == "8U3N"
        assert chain_id == "A"

    def test_handles_multi_char_chain(self):
        pdb_id, chain_id = parse_structure_id("1abc_aa.npz")
        assert pdb_id == "1ABC"
        assert chain_id == "AA"

    def test_rejects_filename_without_underscore(self):
        with pytest.raises(ValueError):
            parse_structure_id("nounderscore.npz")


class TestParseFastaResponse:
    def test_strips_header_and_joins_sequence_lines(self):
        text = ">8U3N.A|Cytochrome P450-SU1|Micromonospora sp.\nMGSSHHHHHH\nSSGLVPRGSH\n"
        assert parse_fasta_response(text) == "MGSSHHHHHHSSGLVPRGSH"

    def test_raises_on_empty_response(self):
        with pytest.raises(ValueError):
            parse_fasta_response("")

    def test_raises_on_response_with_no_sequence_lines(self):
        with pytest.raises(ValueError):
            parse_fasta_response(">8U3N.A|header only\n")


class TestFetchOne:
    def test_returns_sequence_on_200(self):
        session = MagicMock()
        session.get.return_value.status_code = 200
        session.get.return_value.text = ">8U3N.A|desc\nMGSS\n"

        result = fetch_one(session, "8U3N", "A")

        assert result == "MGSS"
        session.get.assert_called_once()
        assert "8U3N.A" in session.get.call_args[0][0]

    def test_raises_on_non_200(self):
        session = MagicMock()
        session.get.return_value.status_code = 404
        session.get.return_value.text = "not found"

        with pytest.raises(RuntimeError):
            fetch_one(session, "8U3N", "A")


class TestLoadExistingIds:
    def test_returns_empty_set_when_file_absent(self, tmp_path):
        assert load_existing_ids(tmp_path / "missing.jsonl") == set()

    def test_returns_structure_ids_from_existing_jsonl(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text(
            json.dumps({"structure_id": "8u3n_a", "sequence": "MGSS"}) + "\n"
            + json.dumps({"structure_id": "1abc_a", "sequence": "AAAA"}) + "\n"
        )
        assert load_existing_ids(path) == {"8u3n_a", "1abc_a"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_fetch_rcsb_structures.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.fetch_rcsb_structures'`

- [ ] **Step 3: Implement**

```python
# src/l40/fetch_rcsb_structures.py
"""Fetches one canonical single-chain sequence per structure from RCSB, for the
structures present in a directory of Boltz-processed .npz shards. This is the
"no MSA augmentation" baseline data source for the L40 ablation: same structure
population as the Boltz side, but without the extra homolog sequences an MSA
shard provides.

Output: a JSONL file, one line per structure:
    {"structure_id": "8u3n_a", "pdb_id": "8U3N", "chain_id": "A", "sequence": "MGSS..."}

Resumable: structures already present in --out are skipped on re-run, same
manifest-based approach as upload_to_s3.py.

Usage:
    .venv-l38/bin/python -m src.l40.fetch_rcsb_structures \
        --npz-dir /path/to/rcsb_processed_msa --out src/l40/data_cache/rcsb_baseline.jsonl \
        --max-structures 2500
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Set, Tuple

import requests

FASTA_URL = "https://www.rcsb.org/fasta/chain/{pdb_id}.{chain_id}"


def parse_structure_id(filename: str) -> Tuple[str, str]:
    stem = filename[:-4] if filename.endswith(".npz") else filename
    if "_" not in stem:
        raise ValueError(f"expected '{{pdbid}}_{{chain}}.npz', got: {filename}")
    pdb_id, chain_id = stem.rsplit("_", 1)
    return pdb_id.upper(), chain_id.upper()


def parse_fasta_response(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines()]
    seq_lines = [line for line in lines if line and not line.startswith(">")]
    if not seq_lines:
        raise ValueError(f"no sequence lines in FASTA response: {text!r}")
    return "".join(seq_lines)


def fetch_one(session, pdb_id: str, chain_id: str) -> str:
    url = FASTA_URL.format(pdb_id=pdb_id, chain_id=chain_id)
    response = session.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"{url} -> HTTP {response.status_code}")
    return parse_fasta_response(response.text)


def load_existing_ids(path: Path) -> Set[str]:
    if not Path(path).exists():
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["structure_id"])
    return ids


def main():
    parser = argparse.ArgumentParser(description="Fetch RCSB baseline sequences matching a Boltz .npz directory")
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-structures", type=int, default=2500)
    parser.add_argument("--sleep-seconds", type=float, default=0.1, help="delay between requests")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))[:args.max_structures]
    print(f"Targeting {len(files)} structures from {args.npz_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done = load_existing_ids(out_path)
    print(f"Already fetched: {len(already_done)}")

    session = requests.Session()
    fetched, failed = 0, []

    with open(out_path, "a") as out_f:
        for filename in files:
            structure_id = filename[:-4]
            if structure_id in already_done:
                continue

            pdb_id, chain_id = parse_structure_id(filename)
            last_err = None
            for attempt in range(args.retries):
                try:
                    sequence = fetch_one(session, pdb_id, chain_id)
                    out_f.write(json.dumps({
                        "structure_id": structure_id, "pdb_id": pdb_id,
                        "chain_id": chain_id, "sequence": sequence,
                    }) + "\n")
                    out_f.flush()
                    fetched += 1
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(args.sleep_seconds * (attempt + 1))

            if last_err is not None:
                failed.append((structure_id, str(last_err)))
                print(f"FAILED: {structure_id}: {last_err}")

            time.sleep(args.sleep_seconds)

            if fetched % 100 == 0 and fetched > 0:
                print(f"  fetched {fetched}/{len(files) - len(already_done)}")

    print(f"Done. Fetched: {fetched}  Failed: {len(failed)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_fetch_rcsb_structures.py -v`
Expected: 9 passed (no real network calls made — `fetch_one`'s tests use a `MagicMock` session)

- [ ] **Step 5: Commit**

```bash
git add src/l40/fetch_rcsb_structures.py tests/l40/test_fetch_rcsb_structures.py
git commit -m "feat(l40): add RCSB baseline sequence fetcher"
```

---

## Task 5: Baseline dataset + dataloaders

**Files:**
- Create: `src/l40/baseline_data.py`
- Test: `tests/l40/test_baseline_data.py`

**Context:** `create_diverse_splits` (already in `msa_data.py`) takes any `Dict[str, List[Dict]]` keyed by filename and returns train/val/test lists of records. Reusing it here — fed the *same* filename-keyed dict shape, built from the same sorted `.npz` filename order — is what pins the split identical to the Boltz side. Each baseline record carries `jsonl_path`/`structure_id` instead of `data_dir`/`filename`, since baseline sequences come from one JSONL, not per-structure `.npz` files.

- [ ] **Step 1: Write the failing tests**

```python
# tests/l40/test_baseline_data.py
import json

import torch

from src.l40.baseline_data import (
    RCSBBaselineDataset,
    create_baseline_dataloaders,
    load_baseline_protein_data,
)
from src.l40.vocab import PAD_TOKEN


def _write_jsonl(path, structure_ids_and_seqs):
    with open(path, "w") as f:
        for sid, seq in structure_ids_and_seqs:
            f.write(json.dumps({"structure_id": sid, "sequence": seq}) + "\n")


class TestLoadBaselineProteinData:
    def test_builds_one_record_per_structure_in_file_order(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("b_a", "AAAA"), ("a_a", "CCCC")])

        file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=10)

        assert list(file_sequences.keys()) == ["b_a", "a_a"]  # preserves JSONL order, NOT sorted
        assert len(file_sequences["b_a"]) == 1

    def test_respects_max_files(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("a", "AAAA"), ("b", "CCCC"), ("c", "GGGG")])

        file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=2)

        assert len(file_sequences) == 2


class TestRCSBBaselineDataset:
    def test_getitem_produces_msadataset_compatible_dict(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3  # 60 residues, all standard AAs
        _write_jsonl(jsonl_path, [("s1", sequence)])
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}
        dataset = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=0)

        item = dataset[0]

        assert set(item.keys()) == {"input_ids", "attention_mask", "labels", "source_file", "seq_length"}
        assert item["input_ids"].shape == (64,)
        assert item["seq_length"] == 60
        assert (item["input_ids"][60:] == PAD_TOKEN).all()

    def test_short_sequence_returns_zero_loss_dummy(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        _write_jsonl(jsonl_path, [("s1", "ARN")])  # length 3 < min 20
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}
        dataset = RCSBBaselineDataset(records=[record], max_length=32)

        item = dataset[0]

        assert item["seq_length"] == 0
        assert (item["labels"] == -100).all()

    def test_fixed_seed_gives_deterministic_masking(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3
        _write_jsonl(jsonl_path, [("s1", sequence)])
        record = {"jsonl_path": str(jsonl_path), "structure_id": "s1"}

        ds1 = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=42)
        ds2 = RCSBBaselineDataset(records=[record], max_length=64, fixed_seed=42)

        assert torch.equal(ds1[0]["input_ids"], ds2[0]["input_ids"])


class TestCreateBaselineDataloaders:
    def test_returns_loaders_with_expected_batch_shape(self, tmp_path):
        jsonl_path = tmp_path / "baseline.jsonl"
        sequence = "ARNDCQEGHILKMFPSTWYV" * 3
        _write_jsonl(jsonl_path, [(f"s{i}", sequence) for i in range(10)])

        train_loader, val_loader, test_loader = create_baseline_dataloaders(
            str(jsonl_path), batch_size=4, max_length=64, max_files=10,
        )

        batch = next(iter(train_loader))
        assert batch["input_ids"].shape[1] == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_baseline_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.baseline_data'`

- [ ] **Step 3: Implement**

```python
# src/l40/baseline_data.py
"""RCSB single-sequence-per-structure dataset — the "no MSA augmentation" arm
of the L40 ablation. Mirrors msa_data.py's MSADataset interface exactly so the
same training loop consumes either data source unmodified.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate
from src.l40.msa_data import create_diverse_splits
from src.l40.vocab import MASK_TOKEN, PAD_TOKEN, sequence_to_ids


class RCSBBaselineDataset(Dataset):
    def __init__(self, records: List[Dict], max_length: int = 512,
                 mask_prob: float = 0.15, fixed_seed: int = None):
        self.records = records
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token = MASK_TOKEN
        self.pad_token = PAD_TOKEN
        self.fixed_seed = fixed_seed
        self._sequence_cache: Dict[str, str] = {}
        self._loaded_paths: set = set()

    def _sequences_for(self, jsonl_path: str) -> Dict[str, str]:
        if jsonl_path not in self._loaded_paths:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        self._sequence_cache[row["structure_id"]] = row["sequence"]
            self._loaded_paths.add(jsonl_path)
        return self._sequence_cache

    def _load_sequence(self, record: Dict) -> np.ndarray:
        sequences = self._sequences_for(record["jsonl_path"])
        seq_str = sequences.get(record["structure_id"], "")
        return sequence_to_ids(seq_str)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        sequence = self._load_sequence(record)

        if len(sequence) < 20:
            dummy = np.zeros(self.max_length, dtype=np.int32)
            labels = np.full(self.max_length, -100, dtype=np.int32)
            attention_mask = np.zeros(self.max_length, dtype=np.float32)
            return {
                'input_ids': torch.tensor(dummy, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
                'labels': torch.tensor(labels, dtype=torch.long),
                'source_file': record['structure_id'],
                'seq_length': 0,
            }

        if self.fixed_seed is not None:
            rng = np.random.RandomState(self.fixed_seed + idx)
        else:
            rng = np.random
        masked_sequence, labels = apply_mlm_masking(sequence, self.mask_prob, self.mask_token, rng)

        masked_sequence = pad_or_truncate(masked_sequence, self.max_length, self.pad_token)
        labels = pad_or_truncate(labels, self.max_length, -100)
        attention_mask = (masked_sequence != self.pad_token).astype(np.float32)

        return {
            'input_ids': torch.tensor(masked_sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long),
            'source_file': record['structure_id'],
            'seq_length': len(sequence),
        }


def load_baseline_protein_data(jsonl_path: str, max_files: int = 1000) -> Dict[str, List[Dict]]:
    """Mirrors msa_data.load_protein_data's dict shape: structure_id -> [record].
    Preserves JSONL line order (the caller is responsible for writing that file
    in the same order as the matching Boltz .npz directory listing, so that
    create_diverse_splits — seeded — produces an identical train/val/test split)."""
    file_sequences: Dict[str, List[Dict]] = {}
    with open(jsonl_path) as f:
        for line in f:
            if len(file_sequences) >= max_files:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            structure_id = row["structure_id"]
            file_sequences[structure_id] = [{
                'jsonl_path': jsonl_path, 'structure_id': structure_id, 'seq_idx': 0,
            }]
    print(f"Indexed {len(file_sequences)} baseline structures from {jsonl_path}")
    return file_sequences


def create_baseline_dataloaders(jsonl_path: str, batch_size: int = 32, max_length: int = 512,
                                 max_files: int = 1000, mask_prob: float = 0.15
                                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    file_sequences = load_baseline_protein_data(jsonl_path, max_files=max_files)

    # max_seqs_per_file_train/val = 1: the baseline has exactly one sequence per
    # structure by construction, so this cap is a no-op — kept explicit for parity
    # with the Boltz side's call signature.
    train_sequences, val_sequences, test_sequences = create_diverse_splits(
        file_sequences, max_seqs_per_file_train=1, max_seqs_per_file_val=1,
    )

    train_dataset = RCSBBaselineDataset(train_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=42)
    val_dataset = RCSBBaselineDataset(val_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=123)
    test_dataset = RCSBBaselineDataset(test_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=456)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_baseline_data.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/l40/baseline_data.py tests/l40/test_baseline_data.py
git commit -m "feat(l40): add RCSB baseline dataset and dataloaders"
```

---

## Task 6: Verify split pinning end-to-end

**Files:**
- Test: `tests/l40/test_split_pinning.py`

This is the correctness-critical guardrail for the whole ablation: given the same filename list, both data-loading paths must assign structures to train/val/test identically.

- [ ] **Step 1: Write the test**

```python
# tests/l40/test_split_pinning.py
import json

from src.l40.baseline_data import load_baseline_protein_data
from src.l40.msa_data import create_diverse_splits, load_protein_data


def test_boltz_and_baseline_produce_identical_file_level_split(tmp_path):
    # Simulate a Boltz npz directory listing.
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    filenames = [f"struct{i}_a.npz" for i in range(20)]
    for name in filenames:
        (npz_dir / name).write_bytes(b"")

    # Baseline JSONL written in the SAME sorted order fetch_rcsb_structures.py would use.
    jsonl_path = tmp_path / "baseline.jsonl"
    with open(jsonl_path, "w") as f:
        for name in sorted(filenames):
            structure_id = name[:-4]
            f.write(json.dumps({"structure_id": structure_id, "sequence": "A" * 40}) + "\n")

    boltz_file_sequences = load_protein_data(str(npz_dir), max_files=20, sequences_per_file=3)
    baseline_file_sequences = load_baseline_protein_data(str(jsonl_path), max_files=20)

    boltz_train, boltz_val, boltz_test = create_diverse_splits(boltz_file_sequences)
    baseline_train, baseline_val, baseline_test = create_diverse_splits(
        baseline_file_sequences, max_seqs_per_file_train=1, max_seqs_per_file_val=1,
    )

    boltz_train_files = {s['filename'] for s in boltz_train}
    baseline_train_files = {s['structure_id'] for s in baseline_train}
    boltz_val_files = {s['filename'] for s in boltz_val}
    baseline_val_files = {s['structure_id'] for s in baseline_val}
    boltz_test_files = {s['filename'] for s in boltz_test}
    baseline_test_files = {s['structure_id'] for s in baseline_test}

    # boltz_*_files are filenames with the .npz suffix (from load_protein_data);
    # baseline_*_files are bare structure_ids (from load_baseline_protein_data) —
    # add the suffix back on for a like-for-like comparison.
    assert boltz_train_files == {f"{sid}.npz" for sid in baseline_train_files}
    assert boltz_val_files == {f"{sid}.npz" for sid in baseline_val_files}
    assert boltz_test_files == {f"{sid}.npz" for sid in baseline_test_files}
```

- [ ] **Step 2: Run to verify it fails first (TDD sanity check), then passes**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_split_pinning.py -v`
Expected: passes on first correct implementation — this test exercises only
already-implemented code (Tasks 1–5), so if it fails, the bug is in how the
filename/structure_id sets are being compared, not missing code.

- [ ] **Step 3: Commit**

```bash
git add tests/l40/test_split_pinning.py
git commit -m "test(l40): pin Boltz and baseline file-level splits to be identical"
```

---

## Task 7: Pilot training script

**Files:**
- Create: `src/l40/train_pilot.py`
- Test: `tests/l40/test_train_pilot.py`

**Files layout for outputs:** `src/l40/pilot_out/{variant}_results.json`

- [ ] **Step 1: Write the failing test** (tiny synthetic data, 1 epoch, real training step — not mocked, matches this repo's "real but small" pilot convention from L35/L37/L38)

```python
# tests/l40/test_train_pilot.py
import json

import numpy as np

from src.l40.train_pilot import run_training


def _write_npz(path, seq_lengths):
    starts, ends, cursor = [], [], 0
    for length in seq_lengths:
        starts.append(cursor)
        cursor += length
        ends.append(cursor)
    sequences = np.array(list(zip(starts, ends)), dtype=[('res_start', 'i4'), ('res_end', 'i4')])
    res_types = np.arange(cursor, dtype='i4') % 20 + 1
    residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])
    np.savez(path, sequences=sequences, residues=residues)


def test_run_training_boltz_variant_produces_finite_metrics(tmp_path):
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    for i in range(12):
        _write_npz(npz_dir / f"s{i}_a.npz", seq_lengths=[40, 40, 40])  # 3 "MSA hits" each

    out_path = tmp_path / "boltz_results.json"
    result = run_training(
        variant="boltz", data_path=str(npz_dir), out_path=str(out_path),
        max_files=12, max_length=48, batch_size=4, epochs=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32,
    )

    assert np.isfinite(result["epochs"][0]["train_loss"])
    assert np.isfinite(result["epochs"][0]["val_loss"])
    assert 0.0 <= result["epochs"][0]["val_accuracy"] <= 1.0
    assert json.loads(out_path.read_text()) == result


def test_run_training_baseline_variant_produces_finite_metrics(tmp_path):
    jsonl_path = tmp_path / "baseline.jsonl"
    sequence = "ARNDCQEGHILKMFPSTWYV" * 3
    with open(jsonl_path, "w") as f:
        for i in range(12):
            f.write(json.dumps({"structure_id": f"s{i}_a", "sequence": sequence}) + "\n")

    out_path = tmp_path / "baseline_results.json"
    result = run_training(
        variant="baseline", data_path=str(jsonl_path), out_path=str(out_path),
        max_files=12, max_length=48, batch_size=4, epochs=1,
        d_model=16, n_layers=1, n_heads=2, d_ff=32,
    )

    assert np.isfinite(result["epochs"][0]["train_loss"])
    assert np.isfinite(result["epochs"][0]["val_loss"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_train_pilot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.train_pilot'`

- [ ] **Step 3: Implement**

```python
# src/l40/train_pilot.py
"""Trains ProteinBERT on either the Boltz-MSA-augmented data or the RCSB
single-sequence baseline, with identical hyperparameters, for the L40 ablation.

Run:
    .venv-l38/bin/python -m src.l40.train_pilot --variant boltz \
        --data-path /path/to/rcsb_processed_msa --max-files 2500 --epochs 5
    .venv-l38/bin/python -m src.l40.train_pilot --variant baseline \
        --data-path src/l40/data_cache/rcsb_baseline.jsonl --max-files 2500 --epochs 5
"""
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from src.l40.baseline_data import create_baseline_dataloaders
from src.l40.model import create_model
from src.l40.msa_data import create_dataloaders

OUT_DIR = Path(__file__).resolve().parent / "pilot_out"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            out = model(input_ids, attention_mask, labels)
            total_loss += out['loss'].item()
            predictions = torch.argmax(out['logits'], dim=-1)
            mask = labels != -100
            correct += ((predictions == labels) & mask).sum().item()
            total += mask.sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    model.train()
    return avg_loss, accuracy


def run_training(variant: str, data_path: str, out_path: str, max_files: int,
                  max_length: int = 512, batch_size: int = 32, epochs: int = 5,
                  mask_prob: float = 0.15, sequences_per_file: int = 5,
                  lr: float = 1e-4, d_model: int = 256, n_layers: int = 6,
                  n_heads: int = 8, d_ff: int = 1024) -> dict:
    device = get_device()
    print(f"[{variant}] device: {device}", flush=True)

    if variant == "boltz":
        train_loader, val_loader, _ = create_dataloaders(
            data_path, batch_size=batch_size, max_length=max_length,
            max_files=max_files, mask_prob=mask_prob, sequences_per_file=sequences_per_file,
        )
    elif variant == "baseline":
        train_loader, val_loader, _ = create_baseline_dataloaders(
            data_path, batch_size=batch_size, max_length=max_length,
            max_files=max_files, mask_prob=mask_prob,
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    model = create_model(
        vocab_size=24, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, max_length=max_length, dropout=0.1,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    result = {"variant": variant, "max_files": max_files, "epochs": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask, labels)
            loss = out['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(f"[{variant}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}", flush=True)

        result["epochs"].append({
            "epoch": epoch + 1, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=["boltz", "baseline"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--out-path", default=None)
    parser.add_argument("--max-files", type=int, default=2500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sequences-per-file", type=int, default=5,
                         help="Boltz variant only: MSA homolog sequences sampled per structure")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1024)
    args = parser.parse_args()

    out_path = args.out_path or str(OUT_DIR / f"{args.variant}_results.json")

    run_training(
        variant=args.variant, data_path=args.data_path, out_path=out_path,
        max_files=args.max_files, max_length=args.max_length, batch_size=args.batch_size,
        epochs=args.epochs, sequences_per_file=args.sequences_per_file, lr=args.lr,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_train_pilot.py -v -s`
Expected: 2 passed (a few seconds each — tiny model, tiny data, CPU/MPS is fine)

- [ ] **Step 5: Commit**

```bash
git add src/l40/train_pilot.py tests/l40/test_train_pilot.py
git commit -m "feat(l40): add shared pilot training script for both ablation variants"
```

---

## Task 8: Comparison script

**Files:**
- Create: `src/l40/compare_results.py`
- Test: `tests/l40/test_compare_results.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/l40/test_compare_results.py
import json

from src.l40.compare_results import compare, load_results


def test_load_results_reads_json(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"variant": "boltz", "epochs": [{"val_accuracy": 0.5}]}))
    result = load_results(str(path))
    assert result["variant"] == "boltz"


def test_compare_reports_final_epoch_delta():
    boltz = {"variant": "boltz", "epochs": [
        {"epoch": 1, "train_loss": 3.0, "val_loss": 2.9, "val_accuracy": 0.10},
        {"epoch": 2, "train_loss": 2.5, "val_loss": 2.4, "val_accuracy": 0.15},
    ]}
    baseline = {"variant": "baseline", "epochs": [
        {"epoch": 1, "train_loss": 3.1, "val_loss": 3.0, "val_accuracy": 0.08},
        {"epoch": 2, "train_loss": 2.7, "val_loss": 2.6, "val_accuracy": 0.11},
    ]}

    result = compare(boltz, baseline)

    assert result["boltz_final_val_accuracy"] == 0.15
    assert result["baseline_final_val_accuracy"] == 0.11
    assert abs(result["val_accuracy_delta"] - 0.04) < 1e-9
    assert abs(result["val_loss_delta"] - (2.6 - 2.4)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_compare_results.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.l40.compare_results'`

- [ ] **Step 3: Implement**

```python
# src/l40/compare_results.py
"""Compares the two L40 pilot training runs and reports the final-epoch delta.

Usage:
    .venv-l38/bin/python -m src.l40.compare_results \
        --boltz src/l40/pilot_out/boltz_results.json \
        --baseline src/l40/pilot_out/baseline_results.json
"""
import argparse
import json


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def compare(boltz: dict, baseline: dict) -> dict:
    boltz_final = boltz["epochs"][-1]
    baseline_final = baseline["epochs"][-1]
    return {
        "boltz_final_train_loss": boltz_final["train_loss"],
        "baseline_final_train_loss": baseline_final["train_loss"],
        "boltz_final_val_loss": boltz_final["val_loss"],
        "baseline_final_val_loss": baseline_final["val_loss"],
        "val_loss_delta": boltz_final["val_loss"] - baseline_final["val_loss"],
        "boltz_final_val_accuracy": boltz_final["val_accuracy"],
        "baseline_final_val_accuracy": baseline_final["val_accuracy"],
        "val_accuracy_delta": boltz_final["val_accuracy"] - baseline_final["val_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boltz", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    boltz = load_results(args.boltz)
    baseline = load_results(args.baseline)
    result = compare(boltz, baseline)

    print(json.dumps(result, indent=2))
    print()
    print(f"val_accuracy: boltz={result['boltz_final_val_accuracy']:.4f}  "
          f"baseline={result['baseline_final_val_accuracy']:.4f}  "
          f"delta={result['val_accuracy_delta']:+.4f}")
    print(f"val_loss:     boltz={result['boltz_final_val_loss']:.4f}  "
          f"baseline={result['baseline_final_val_loss']:.4f}  "
          f"delta={result['val_loss_delta']:+.4f}  (negative = boltz better)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-l38/bin/python -m pytest tests/l40/test_compare_results.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/l40/compare_results.py tests/l40/test_compare_results.py
git commit -m "feat(l40): add pilot results comparison script"
```

---

## Task 9: Pre-registered protocol doc

**Files:**
- Create: `docs/L40_PROTOCOL.md`

- [ ] **Step 1: Write the doc**

```markdown
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
on MPS (no CUDA available on this machine) or the smallest available EC2 GPU
if MPS throughput is too slow. This is a first-pass signal check, not a
scaled/definitive result.

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
```

- [ ] **Step 2: Commit**

```bash
git add docs/L40_PROTOCOL.md
git commit -m "docs(l40): pre-register the Boltz-MSA-augmentation ablation pilot protocol"
```

---

## Task 10: Run the real pilot

**Not a code task — this is the actual experiment execution**, after all of Tasks 1–9 are built and tested.

- [ ] **Step 1:** Download a pilot-sized slice of Boltz's `.npz` data via HTTP Range request (confirmed working: a 100MB range yields ~170 complete `.npz` members via `tar -x --strip-components=1`; scale the range up for ~2500 structures, expect roughly 1.5GB based on the observed ~590KB/file average — round up and over-fetch since the tail file in any range is truncated and discarded).

```bash
mkdir -p data/l40_pilot/npz
curl -s -r 0-1600000000 "https://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar" \
  | tar -x --strip-components=1 -C data/l40_pilot/npz 2>/dev/null
ls data/l40_pilot/npz/*.npz | wc -l   # confirm >= 2500
```

- [ ] **Step 2:** Fetch matching RCSB baseline sequences

```bash
.venv-l38/bin/python -m src.l40.fetch_rcsb_structures \
  --npz-dir data/l40_pilot/npz --out data/l40_pilot/rcsb_baseline.jsonl --max-structures 2500
```

- [ ] **Step 3:** Train both variants with identical hyperparameters

```bash
.venv-l38/bin/python -m src.l40.train_pilot --variant boltz \
  --data-path data/l40_pilot/npz --max-files 2500 --epochs 5 --batch-size 32 \
  --sequences-per-file 5

.venv-l38/bin/python -m src.l40.train_pilot --variant baseline \
  --data-path data/l40_pilot/rcsb_baseline.jsonl --max-files 2500 --epochs 5 --batch-size 32
```

- [ ] **Step 4:** Compare

```bash
.venv-l38/bin/python -m src.l40.compare_results \
  --boltz src/l40/pilot_out/boltz_results.json \
  --baseline src/l40/pilot_out/baseline_results.json
```

- [ ] **Step 5:** Append the real numbers + verdict to `docs/L40_PROTOCOL.md` under a
new `## Result (YYYY-MM-DD)` section, and commit.

```bash
git add docs/L40_PROTOCOL.md src/l40/pilot_out/
git commit -m "docs(l40): record pilot ablation result"
```

---

## Self-Review Notes

- **Spec coverage:** brainstormed design called for (1) RCSB fetcher [Task 4],
  (2) baseline dataset adapter [Task 5], (3) shared training entrypoint
  [Task 7], (4) comparison/report script [Task 8], plus the structure-ID
  pinning invariant [Task 6] and pre-registration doc [Task 9] added during
  planning. All covered.
- **Split-pinning is the single highest-risk correctness point** — Task 6
  exists specifically to catch a silent mismatch before any GPU time is spent.
- Task 10 has no automated test (it's real network + real training) —
  correctness for that step rests entirely on Tasks 1–9's unit/integration
  tests plus the live RCSB-endpoint and Boltz-range-request checks already
  performed during design (both confirmed working before this plan was written).

**Execution-time findings (fixed during Task 7, not caught by planning):**
- `nn.TransformerEncoder`'s padding-mask fast path calls
  `aten::_nested_tensor_from_mask_left_aligned`, unimplemented on MPS —
  `train_pilot.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` at module import time,
  and `tests/l40/conftest.py` sets the same env var even earlier, since
  pytest's alphabetical collection order imports `torch` (via
  `test_baseline_data.py`) before `train_pilot.py` gets a chance to.
- `run_training` didn't seed `torch` before model construction — with a tiny
  pilot-scale model (`d_model=16`, 2 heads) this occasionally produced a NaN
  training loss from an unlucky init, and more importantly meant Model A and
  Model B could start from different initial weights, confounding the
  ablation. Fixed: `torch.manual_seed(seed)` (default 0, same value for both
  variants) right before `create_model(...)`.
