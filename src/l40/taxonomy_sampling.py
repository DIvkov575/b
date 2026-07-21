"""Taxonomy-diversity-aware homolog sampling.

Matches Boltz's real use of the 'taxonomy' field (verified against Boltz's
source, boltz.model.modules.trunkv2.construct_paired_msa): taxonomy is a
selection/grouping key, never a model input feature. Here it's used to pick
a taxonomically diverse subset of homologs per structure, rather than the
arbitrary first-N pick msa_data.py's create_diverse_splits uses.
"""
from typing import List

import numpy as np


def sample_diverse_homologs(sequences: np.ndarray, n: int, rng) -> List[np.ndarray]:
    """sequences is the .npz 'sequences' structured array for one structure
    (fields: seq_idx, taxonomy, res_start, res_end, del_start, del_end), with
    the query at some row where taxonomy == -1. Returns up to n rows: the
    query first, then up to n-1 homologs preferring distinct taxonomy IDs
    (one per taxon before any taxon repeats), falling back to repeats only
    if there are fewer distinct homolog taxa than needed. rng must expose
    .permutation(x) -> shuffled array (np.random.RandomState satisfies this)."""
    query_mask = sequences['taxonomy'] == -1
    query_rows = sequences[query_mask]
    homolog_rows = sequences[~query_mask]

    selected = [query_rows[0]] if len(query_rows) > 0 else []
    n_homologs_needed = n - len(selected)

    if n_homologs_needed <= 0 or len(homolog_rows) == 0:
        return selected

    taxa = homolog_rows['taxonomy']
    unique_taxa = rng.permutation(np.unique(taxa))

    picked_indices = []
    for taxon in unique_taxa:
        if len(picked_indices) >= n_homologs_needed:
            break
        candidates = np.where(taxa == taxon)[0]
        picked_indices.append(rng.permutation(candidates)[0])

    remaining_needed = n_homologs_needed - len(picked_indices)
    if remaining_needed > 0:
        already_picked = set(picked_indices)
        pool = [i for i in range(len(homolog_rows)) if i not in already_picked]
        if pool:
            extra = rng.permutation(pool)[:remaining_needed]
            picked_indices.extend(extra.tolist())
            remaining_needed -= len(extra)

    # Every distinct homolog has now been used at least once (or there
    # weren't enough homologs to fill the pool above) — any further slots
    # are filled by repeating from the full homolog set.
    if remaining_needed > 0 and len(homolog_rows) > 0:
        repeats = rng.randint(0, len(homolog_rows), size=remaining_needed)
        picked_indices.extend(repeats.tolist())

    for idx in picked_indices:
        selected.append(homolog_rows[idx])

    return selected
