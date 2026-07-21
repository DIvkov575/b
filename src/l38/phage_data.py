"""L39 -- phage/viral protein ESM-2 fine-tune: FASTA parsing + train/eval split.

ESM-2's training corpus (UniRef) is documented to underrepresent viral/phage
sequences (Sawhney et al. 2025, PeerJ, DOI 10.7717/peerj.19919 -- "viral
proteins... dark matter" in PLM training data). This module loads the
Caudoviricetes (phage) sequences pulled from UniProt and a general-protein
comparison set, for a masked-LM fine-tune + perplexity-gap eval.
"""
import random
from pathlib import Path
from typing import List, Tuple


def parse_fasta(path: Path) -> List[str]:
    """Return the list of sequences in a FASTA file (headers dropped)."""
    sequences = []
    current: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    return sequences


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequences(sequences: List[str], min_len: int = 20, max_len: int = 512) -> List[str]:
    """Drop sequences with non-standard residues or out-of-range length."""
    cleaned = []
    for seq in sequences:
        if not (min_len <= len(seq) <= max_len):
            continue
        if not set(seq) <= VALID_AA:
            continue
        cleaned.append(seq)
    return cleaned


def train_eval_split(sequences: List[str], eval_frac: float = 0.1, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Deterministic shuffle-then-split. eval_frac must be in (0, 1)."""
    if not (0.0 < eval_frac < 1.0):
        raise ValueError("eval_frac must be in (0, 1)")
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_frac))
    return shuffled[n_eval:], shuffled[:n_eval]
