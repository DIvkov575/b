"""L39 v3 -- real PhaVIP/ESM-PVP-lineage dataset loading.

pvp.fa / non_pvp.fa are the real, full RefSeq-derived phage-protein pool
hosted by PhaVIP's webserver (phage.ee.cityu.edu.hk/download) -- the same
data lineage ESM-PVP's paper cites as its training/test source (via PhaVIP,
Shang et al. 2023). This is NOT ESM-PVP's exact time-split (pre/post
Dec-2020 by RefSeq release date), since no release-date metadata is present
in these files -- but it is the real, correctly-sourced labeled pool, a
substantial upgrade over the earlier UniProt-keyword approximation
(5,990 sequences) used in the original L39 virion_eval.py.
"""
from pathlib import Path
from typing import List, Tuple

from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split

DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "phavip_real"


def load_pvp_labeled_dataset(min_len: int = 20, max_len: int = 1024) -> Tuple[List[str], List[int]]:
    """Returns (sequences, labels) with label 1=PVP (virion protein), 0=non-PVP.
    Deduplicates any sequence appearing in both files (drops from both, as in
    virion_eval.py's load_labeled_dataset, to avoid a contradictory label)."""
    pvp = clean_sequences(parse_fasta(DATA_DIR / "pvp.fa"), min_len=min_len, max_len=max_len)
    non_pvp = clean_sequences(parse_fasta(DATA_DIR / "non_pvp.fa"), min_len=min_len, max_len=max_len)

    pvp_set, non_pvp_set = set(pvp), set(non_pvp)
    overlap = pvp_set & non_pvp_set
    pvp = [s for s in pvp if s not in overlap]
    non_pvp = [s for s in non_pvp if s not in overlap]

    sequences = pvp + non_pvp
    labels = [1] * len(pvp) + [0] * len(non_pvp)
    return sequences, labels


def load_mlm_corpus(eval_frac: float = 0.05, seed: int = 0, min_len: int = 20, max_len: int = 1024) -> Tuple[List[str], List[str]]:
    """Union of pvp.fa + non_pvp.fa as an unlabeled MLM fine-tuning corpus
    (both are real phage proteins; the label distinction doesn't matter for
    masked-LM pretraining). Held-out eval slice is disjoint from train by
    construction (train_eval_split's shuffle-then-split)."""
    sequences, _ = load_pvp_labeled_dataset(min_len=min_len, max_len=max_len)
    return train_eval_split(sequences, eval_frac=eval_frac, seed=seed)
