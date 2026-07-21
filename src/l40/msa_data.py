"""MSA .npz data loading, MLM masking, splitting, and DataLoader construction.

Ported from PFold (github.com/DIvkov575/PFold, commit f44eecc) data.py.
Expects Boltz-style pre-processed RCSB MSA .npz shards, each containing a
`sequences` structured array (with `res_start`/`res_end` fields) and a
`residues` structured array (with a `res_type` field) indexing into it.
"""
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate
from src.l40.vocab import MASK_TOKEN, PAD_TOKEN


class MSADataset(Dataset):
    def __init__(self, records: List[Dict], max_length: int = 512,
                 mask_prob: float = 0.15, fixed_seed: int = None):
        # Each record is lightweight: {data_dir, filename, seq_idx, length}
        # The actual sequence array is NOT stored — loaded on demand in __getitem__.
        self.records = records
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.mask_token = MASK_TOKEN
        self.pad_token = PAD_TOKEN
        self.fixed_seed = fixed_seed

    def _load_sequence(self, record: Dict) -> np.ndarray:
        filepath = os.path.join(record['data_dir'], record['filename'])
        data = np.load(filepath)
        sequences = data['sequences']
        if record['seq_idx'] >= len(sequences):
            return np.array([], dtype=np.int32)
        residues = data['residues']['res_type']
        seq_info = sequences[record['seq_idx']]
        return residues[seq_info['res_start']:seq_info['res_end']].astype(np.int32)

    def _apply_mlm_masking(self, sequence: np.ndarray, seq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.fixed_seed is not None:
            rng = np.random.RandomState(self.fixed_seed + seq_idx)
        else:
            rng = np.random
        return apply_mlm_masking(sequence, self.mask_prob, self.mask_token, rng)

    def _pad_sequence(self, sequence: np.ndarray, is_labels: bool = False) -> np.ndarray:
        pad_value = -100 if is_labels else self.pad_token
        return pad_or_truncate(sequence, self.max_length, pad_value)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        sequence = self._load_sequence(record)

        # Skip sequences that are too short — return an all-masked dummy that contributes
        # zero loss (all labels = -100, attention_mask = 0).
        if len(sequence) < 20:
            dummy = np.zeros(self.max_length, dtype=np.int32)
            labels = np.full(self.max_length, -100, dtype=np.int32)
            attention_mask = np.zeros(self.max_length, dtype=np.float32)
            return {
                'input_ids': torch.tensor(dummy, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
                'labels': torch.tensor(labels, dtype=torch.long),
                'source_file': record['filename'],
                'seq_length': 0,
            }

        masked_sequence, labels = self._apply_mlm_masking(sequence, idx)
        masked_sequence = self._pad_sequence(masked_sequence, is_labels=False)
        labels = self._pad_sequence(labels, is_labels=True)
        attention_mask = (masked_sequence != self.pad_token).astype(np.float32)

        return {
            'input_ids': torch.tensor(masked_sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels, dtype=torch.long),
            'source_file': record['filename'],
            'seq_length': len(sequence),
        }


def load_protein_data(data_dir: str, max_files: int = 1000,
                       sequences_per_file: int = 10, min_seq_length: int = 20) -> Dict[str, List[Dict]]:
    """Returns dict mapping filename -> list of lightweight records.
    No files are opened — records are built from the directory listing alone.
    Sequences are loaded on demand in MSADataset.__getitem__."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))[:max_files]
    file_sequences = {
        f: [{'data_dir': data_dir, 'filename': f, 'seq_idx': i, 'length': None}
            for i in range(sequences_per_file)]
        for f in files
    }
    print(f"Indexed {len(file_sequences)} files (no scan — sequences load on demand)")
    return file_sequences


def create_diverse_splits(file_sequences: Dict[str, List[Dict]],
                           train_file_ratio: float = 0.7,
                           val_file_ratio: float = 0.15,
                           max_seqs_per_file_train: int = 5,
                           max_seqs_per_file_val: int = 2) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    all_files = list(file_sequences.keys())
    random.seed(42)  # For reproducible splits
    random.shuffle(all_files)

    # Split files
    n_train_files = int(len(all_files) * train_file_ratio)
    n_val_files = int(len(all_files) * val_file_ratio)

    train_files = all_files[:n_train_files]
    val_files = all_files[n_train_files:n_train_files + n_val_files]

    test_files = all_files[n_train_files + n_val_files:]  # leftovers

    print(f"file splits: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # training set
    train_sequences = []
    for filename in train_files:
        seqs = file_sequences[filename]
        if len(seqs) > max_seqs_per_file_train:
            selected_seqs = random.sample(seqs, max_seqs_per_file_train)
        else:
            selected_seqs = seqs

        # All selected sequences go to training
        train_sequences.extend(selected_seqs)

    # validation set
    val_sequences = []
    for filename in val_files:
        seqs = file_sequences[filename]

        num_val_seqs = min(len(seqs), max_seqs_per_file_val)
        if num_val_seqs > 1:
            selected = random.sample(seqs, num_val_seqs)
        else:
            selected = seqs
        val_sequences.extend(selected)

    # test set: single sequence per test file
    test_sequences = []
    for filename in test_files:
        seqs = file_sequences[filename]
        if seqs:
            # Take the longest sequence as most representative
            best_seq = seqs[0]
            test_sequences.append(best_seq)

    print(f"sequences: {len(train_sequences)} train, {len(val_sequences)} val, "
          f"{len(test_sequences)} test")

    return train_sequences, val_sequences, test_sequences


def create_dataloaders(data_dir: str, batch_size: int = 32, max_length: int = 512,
                        max_files: int = 1000, mask_prob: float = 0.15,
                        sequences_per_file: int = 10) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    create training, validation, and test dataloaders with proper data splitting.
    Returns: (train_loader, val_loader, test_loader)
    """

    file_sequences: Dict[str, List[Dict]] = load_protein_data(
        data_dir,
        max_files=max_files,
        sequences_per_file=sequences_per_file
    )

    train_sequences, val_sequences, test_sequences = create_diverse_splits(file_sequences)  # lists of seqs

    train_dataset = MSADataset(train_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=42)
    val_dataset = MSADataset(val_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=123)
    test_dataset = MSADataset(test_sequences, max_length=max_length, mask_prob=mask_prob, fixed_seed=456)

    # Each worker loads .npz files on demand via FastFile.
    # Use all available CPUs and a high prefetch_factor to keep S3 reads in flight.
    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True, prefetch_factor=4, persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"SEQUENC SPLIT: {len(train_sequences)}, {len(val_sequences)}, {len(test_sequences)} ")

    train_files = set(seq['filename'] for seq in train_sequences)
    val_files = set(seq['filename'] for seq in val_sequences)
    test_files = set(seq['filename'] for seq in test_sequences)

    print(f"Files: Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    return train_loader, val_loader, test_loader
