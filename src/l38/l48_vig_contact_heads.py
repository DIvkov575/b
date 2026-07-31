"""L48 Task A: redo Vig et al.'s "BERTology Meets Biology" (ICLR 2021,
arXiv:2006.15222) contact-map attention-head finding as a real CAUSAL test.

Vig et al. found specific attention heads in protein LMs (best result:
ProtBert-BFD, head 12-4 style finding, 63.2% of that head's attention on
residue pairs in physical contact vs. background rate) but state explicitly:
"all of the above analyses are purely associative and do not attempt to
establish a causal link." A dedicated literature search (2026-07-30)
confirmed nobody has gone back and ablated these specific heads to test
whether contact-prediction actually depends on them -- this fills that gap.

Two-stage design:
1. REPLICATE the correlational finding first, on real PDB structures, using
   the actual model Vig's paper found the strongest result on
   (Rostlab/prot_bert_bfd) -- confirms we're looking at the right
   model/heads before spending effort on a causal test of the wrong thing.
2. CAUSAL TEST: ablate (zero out) the most contact-aligned head's output
   and measure whether a real downstream task -- masked-residue prediction
   accuracy, conditioned on distant contacting residues -- degrades more
   than ablating a random control head of matched norm.

Pure-math/data-plumbing pieces (contact map from PDB coordinates, head
enrichment scoring) are separated from model-forward-pass pieces so the
former can be unit tested without downloading a multi-GB model.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

STANDARD_AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CONTACT_DISTANCE_THRESHOLD = 8.0  # angstroms, Cα-Cα -- standard convention
# in protein structure/contact-prediction literature (e.g. CASP contact
# prediction assessment), also the threshold used in Rao et al./the TAPE
# benchmark that Vig et al.'s own contact-prediction evaluation builds on.
MIN_SEQUENCE_SEPARATION = 6  # exclude near-diagonal residue pairs (i,j with
# |i-j| < 6) -- these are almost always "in contact" trivially due to chain
# connectivity, not because of any interesting long-range structural
# feature. Standard convention in the contact-prediction literature
# (matches TAPE/CASP's short/medium/long-range contact categorization,
# using "long-range" as the interesting regime); confirmed via direct
# computation that using min_sep=0 on a compact protein like ubiquitin
# gives a background contact rate several times higher than Vig et al.'s
# reported 1.3% (likely due to exactly this near-diagonal inflation).


def extract_sequence_and_contact_map(pdb_path: Path, chain_id: str = None) -> Tuple[str, np.ndarray]:
    """Parse a PDB file, return (sequence, boolean contact map) for the
    first (or specified) chain. Contact map excludes near-diagonal pairs
    per MIN_SEQUENCE_SEPARATION and is symmetric.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    model = structure[0]
    chain = model[chain_id] if chain_id else next(model.get_chains())

    residues = [r for r in chain if "CA" in r and r.get_resname() in STANDARD_AA_3TO1]
    sequence = "".join(STANDARD_AA_3TO1[r.get_resname()] for r in residues)
    coords = np.array([r["CA"].get_coord() for r in residues])

    n = len(coords)
    dists = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sep_mask = np.abs(ii - jj) >= MIN_SEQUENCE_SEPARATION
    contact_map = (dists < CONTACT_DISTANCE_THRESHOLD) & sep_mask

    return sequence, contact_map


def contact_background_rate(contact_map: np.ndarray) -> float:
    """Fraction of all (sequence-separation-eligible) residue pairs that
    are in contact -- the baseline rate a random/uninformed head would be
    expected to hit by chance."""
    n = contact_map.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eligible = np.abs(ii - jj) >= MIN_SEQUENCE_SEPARATION
    if eligible.sum() == 0:
        raise ValueError("no eligible residue pairs (sequence too short for MIN_SEQUENCE_SEPARATION)")
    return float(contact_map.sum()) / float(eligible.sum())


def head_contact_enrichment(attention_weights: np.ndarray, contact_map: np.ndarray) -> Dict[str, float]:
    """Given one head's attention matrix [seq_len, seq_len] and a boolean
    contact map of the SAME shape, compute what fraction of this head's
    total attention mass lands on contacting pairs (Vig et al.'s exact
    metric), restricted to sequence-separation-eligible pairs. Also
    returns the background rate for direct enrichment-ratio comparison.
    """
    if attention_weights.shape != contact_map.shape:
        raise ValueError(
            f"attention_weights shape {attention_weights.shape} must match contact_map shape {contact_map.shape}"
        )
    n = contact_map.shape[0]
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eligible = np.abs(ii - jj) >= MIN_SEQUENCE_SEPARATION

    total_eligible_attention = attention_weights[eligible].sum()
    if total_eligible_attention <= 0:
        return {"attention_on_contacts_fraction": 0.0, "background_rate": contact_background_rate(contact_map), "enrichment_ratio": 0.0}

    attention_on_contacts = attention_weights[eligible & contact_map].sum()
    fraction = float(attention_on_contacts / total_eligible_attention)
    background = contact_background_rate(contact_map)
    enrichment = fraction / background if background > 0 else float("inf")

    return {
        "attention_on_contacts_fraction": fraction,
        "background_rate": background,
        "enrichment_ratio": enrichment,
    }
