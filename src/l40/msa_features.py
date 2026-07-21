"""MSA-derived features from Boltz's real .npz format: deletion counts and
per-position amino-acid profile. Transforms verified against Boltz's own
source (boltz.data.feature.featurizerv2): deletion_value = pi/2 * arctan(d/3),
has_deletion = raw_count > 0 (computed before the arctan transform).
"""
from typing import Tuple

import numpy as np


def compute_deletion_features(sequence_record, deletions: np.ndarray,
                                seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """sequence_record is one row of the .npz 'sequences' structured array
    (fields: seq_idx, taxonomy, res_start, res_end, del_start, del_end).
    deletions is the full .npz 'deletions' array (fields: res_idx, deletion).
    Returns (has_deletion, deletion_value), both shape (seq_length,)."""
    raw_counts = np.zeros(seq_length, dtype=np.float64)
    del_start, del_end = sequence_record['del_start'], sequence_record['del_end']
    for row in deletions[del_start:del_end]:
        res_idx = int(row['res_idx'])
        if 0 <= res_idx < seq_length:
            raw_counts[res_idx] = row['deletion']

    has_deletion = raw_counts > 0
    deletion_value = np.pi / 2 * np.arctan(raw_counts / 3)
    return has_deletion, deletion_value


def compute_profile(sequences: np.ndarray, residues: np.ndarray, seq_length: int,
                     vocab_size: int) -> np.ndarray:
    """Per-position amino-acid frequency across all given sequences (rows of the
    .npz 'sequences' structured array), aligned by column position 0..seq_length-1.
    Returns shape (seq_length, vocab_size), each row summing to 1.0."""
    res_types = residues['res_type']
    counts = np.zeros((seq_length, vocab_size), dtype=np.float64)

    for seq in sequences:
        start, end = seq['res_start'], seq['res_end']
        seq_res_types = res_types[start:end]
        for pos in range(min(seq_length, len(seq_res_types))):
            counts[pos, seq_res_types[pos]] += 1.0

    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals
