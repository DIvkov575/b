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
