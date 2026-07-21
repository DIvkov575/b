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
