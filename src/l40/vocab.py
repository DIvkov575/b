"""Amino-acid vocabulary used by the ported PFold MSA data pipeline.

Ported from PFold (github.com/DIvkov575/PFold, commit f44eecc) config.py.
"""
import numpy as np

AA_VOCAB = {
    'PAD': 0,   # Padding token
    'A': 1,     # Alanine
    'R': 2,     # Arginine
    'N': 3,     # Asparagine
    'D': 4,     # Aspartic acid
    'C': 5,     # Cysteine
    'Q': 6,     # Glutamine
    'E': 7,     # Glutamic acid
    'G': 8,     # Glycine
    'H': 9,     # Histidine
    'I': 10,    # Isoleucine
    'L': 11,    # Leucine
    'K': 12,    # Lysine
    'M': 13,    # Methionine
    'F': 14,    # Phenylalanine
    'P': 15,    # Proline
    'S': 16,    # Serine
    'T': 17,    # Threonine
    'W': 18,    # Tryptophan
    'Y': 19,    # Tyrosine
    'V': 20,    # Valine
    'X': 21,    # Unknown/non-standard
    'GAP': 22,  # Gap in alignment
    'MASK': 23,  # Mask token for MLM
}

VOCAB_TO_AA = {v: k for k, v in AA_VOCAB.items()}

VOCAB_SIZE = 24
MASK_TOKEN = AA_VOCAB['MASK']
PAD_TOKEN = AA_VOCAB['PAD']

_LETTER_TO_ID = {k: v for k, v in AA_VOCAB.items() if len(k) == 1}
_UNKNOWN_ID = AA_VOCAB['X']


def sequence_to_ids(seq: str) -> np.ndarray:
    """Maps a one-letter amino-acid sequence to AA_VOCAB integer ids.
    Unknown/non-standard letters (e.g. U, Z, B, O, J) map to 'X'."""
    return np.array(
        [_LETTER_TO_ID.get(c.upper(), _UNKNOWN_ID) for c in seq],
        dtype=np.int32,
    )
