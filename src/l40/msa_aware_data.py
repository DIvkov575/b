"""Dataset for the full-ablation MSA-aware model: per structure, loads the
query sequence plus a taxonomically-diverse sample of homologs, computes
deletion/profile features from ALL available homologs (not just the sampled
subset -- matches Boltz's own featurizer, which computes profile from the
full MSA before any subsampling), and returns tensors shaped for
MSAAwareProteinBERT's forward pass. Only the query row is MLM-masked; homolog
rows keep their real tokens (they're context, not training targets).
"""
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.l40.mlm_common import apply_mlm_masking, pad_or_truncate
from src.l40.msa_data import create_diverse_splits
from src.l40.msa_features import compute_deletion_features, compute_profile
from src.l40.taxonomy_sampling import sample_diverse_homologs
from src.l40.vocab import MASK_TOKEN, PAD_TOKEN, VOCAB_SIZE


def load_msa_aware_protein_data(data_dir: str, max_files: int = 1000) -> Dict[str, List[Dict]]:
    """Mirrors msa_data.load_protein_data's dict shape, but one record per
    file (the dataset itself samples homologs at __getitem__ time, since that
    needs the query's own MSA depth, unknown from a directory listing alone)."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))[:max_files]
    file_sequences = {
        f: [{'data_dir': data_dir, 'filename': f}]
        for f in files
    }
    print(f"Indexed {len(file_sequences)} files for MSA-aware loading")
    return file_sequences


class MSAAwareDataset(Dataset):
    def __init__(self, records: List[Dict], max_length: int = 512, msa_depth: int = 8,
                 mask_prob: float = 0.15, fixed_seed: int = None,
                 vocab_size: int = VOCAB_SIZE):
        self.records = records
        self.max_length = max_length
        self.msa_depth = msa_depth
        self.mask_prob = mask_prob
        self.fixed_seed = fixed_seed
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.records)

    def _dummy_item(self) -> Dict[str, torch.Tensor]:
        return {
            'msa_tokens': torch.zeros(self.msa_depth, self.max_length, dtype=torch.long),
            'has_deletion': torch.zeros(self.msa_depth, self.max_length),
            'deletion_value': torch.zeros(self.msa_depth, self.max_length),
            'profile': torch.zeros(self.max_length, self.vocab_size),
            'deletion_mean': torch.zeros(self.max_length),
            'attention_mask': torch.zeros(self.max_length),
            'labels': torch.full((self.max_length,), -100, dtype=torch.long),
            'source_file': '',
            'seq_length': 0,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        filepath = os.path.join(record['data_dir'], record['filename'])
        data = np.load(filepath)
        all_sequences = data['sequences']
        deletions = data['deletions']
        res_types = data['residues']['res_type']

        query_mask = all_sequences['taxonomy'] == -1
        if not query_mask.any():
            return self._dummy_item()
        query_row = all_sequences[query_mask][0]
        seq_len = int(query_row['res_end'] - query_row['res_start'])

        if seq_len < 20:
            return self._dummy_item()

        rng = np.random.RandomState(self.fixed_seed + idx) if self.fixed_seed is not None else np.random

        selected_rows = sample_diverse_homologs(all_sequences, n=self.msa_depth, rng=rng)

        # Profile/deletion_mean use ALL homologs regardless of msa_depth
        # sampling, matching Boltz's real featurizer (profile is computed
        # from the full MSA before any subsampling).
        profile = compute_profile(all_sequences, data['residues'], seq_len, self.vocab_size)
        all_deletion_values = []
        for row in all_sequences:
            row_len = int(row['res_end'] - row['res_start'])
            if row_len != seq_len:
                continue
            _, del_val = compute_deletion_features(row, deletions, seq_len)
            all_deletion_values.append(del_val)
        deletion_mean = (np.mean(all_deletion_values, axis=0) if all_deletion_values
                          else np.zeros(seq_len))

        msa_tokens_rows, has_deletion_rows, deletion_value_rows = [], [], []
        for row in selected_rows:
            row_seq = res_types[row['res_start']:row['res_end']].astype(np.int32)
            has_del, del_val = compute_deletion_features(row, deletions, seq_len)
            msa_tokens_rows.append(pad_or_truncate(row_seq, self.max_length, PAD_TOKEN))
            has_deletion_rows.append(pad_or_truncate(has_del.astype(np.float32), self.max_length, 0.0))
            deletion_value_rows.append(pad_or_truncate(del_val.astype(np.float32), self.max_length, 0.0))

        # Pad the MSA-depth axis up to msa_depth by repeating already-selected
        # rows (only reached when the structure has fewer homologs than
        # msa_depth -- sample_diverse_homologs already returns as many
        # distinct-then-repeated rows as it can, this covers the remainder).
        while len(msa_tokens_rows) < self.msa_depth:
            repeat_idx = (len(msa_tokens_rows) - 1) % len(selected_rows)
            msa_tokens_rows.append(msa_tokens_rows[repeat_idx])
            has_deletion_rows.append(has_deletion_rows[repeat_idx])
            deletion_value_rows.append(deletion_value_rows[repeat_idx])

        query_seq = res_types[query_row['res_start']:query_row['res_end']].astype(np.int32)
        if self.fixed_seed is not None:
            mask_rng = np.random.RandomState(self.fixed_seed + idx)
        else:
            mask_rng = np.random
        masked_query, labels = apply_mlm_masking(query_seq, self.mask_prob, MASK_TOKEN, mask_rng)
        msa_tokens_rows[0] = pad_or_truncate(masked_query, self.max_length, PAD_TOKEN)

        attention_mask = (msa_tokens_rows[0] != PAD_TOKEN).astype(np.float32)
        labels_padded = pad_or_truncate(labels, self.max_length, -100)
        profile_padded = np.zeros((self.max_length, self.vocab_size), dtype=np.float32)
        profile_padded[:min(seq_len, self.max_length)] = profile[:self.max_length]
        deletion_mean_padded = pad_or_truncate(
            np.asarray(deletion_mean, dtype=np.float32), self.max_length, 0.0,
        )

        return {
            'msa_tokens': torch.tensor(np.stack(msa_tokens_rows), dtype=torch.long),
            'has_deletion': torch.tensor(np.stack(has_deletion_rows), dtype=torch.float),
            'deletion_value': torch.tensor(np.stack(deletion_value_rows), dtype=torch.float),
            'profile': torch.tensor(profile_padded, dtype=torch.float),
            'deletion_mean': torch.tensor(deletion_mean_padded, dtype=torch.float),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float),
            'labels': torch.tensor(labels_padded, dtype=torch.long),
            'source_file': record['filename'],
            'seq_length': seq_len,
        }


def create_msa_aware_dataloaders(data_dir: str, batch_size: int = 32, max_length: int = 512,
                                  max_files: int = 1000, msa_depth: int = 8,
                                  mask_prob: float = 0.15
                                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Same file-level split as msa_data.create_dataloaders (same
    create_diverse_splits seed) so the full-ablation arms and the earlier
    training-volume-ablation arms compare against an identical structure
    population and train/val/test assignment."""
    file_sequences = load_msa_aware_protein_data(data_dir, max_files=max_files)

    train_sequences, val_sequences, test_sequences = create_diverse_splits(
        file_sequences, max_seqs_per_file_train=1, max_seqs_per_file_val=1,
    )

    train_dataset = MSAAwareDataset(train_sequences, max_length=max_length, msa_depth=msa_depth,
                                     mask_prob=mask_prob, fixed_seed=42)
    val_dataset = MSAAwareDataset(val_sequences, max_length=max_length, msa_depth=msa_depth,
                                   mask_prob=mask_prob, fixed_seed=123)
    test_dataset = MSAAwareDataset(test_sequences, max_length=max_length, msa_depth=msa_depth,
                                    mask_prob=mask_prob, fixed_seed=456)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
