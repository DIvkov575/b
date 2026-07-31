"""L48 Task A, Stage 1: replicate Vig et al.'s contact-map attention-head
finding on real PDB structures, using Rostlab/prot_bert_bfd -- the exact
model their paper found the STRONGEST result on (63.2% attention-on-
contacts for their best head, vs. their other 4 models' 44-59%).

Must replicate the correlational finding on our own data/model setup
BEFORE spending effort on causally testing it -- confirms we're looking at
a real, reproducible phenomenon and not chasing a head that only appeared
in their exact original data slice.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from src.l38.l48_vig_contact_heads import (
    extract_sequence_and_contact_map,
    head_contact_enrichment,
)

MODEL_NAME = "Rostlab/prot_bert_bfd"
PDB_DIR = Path(__file__).resolve().parent / "data_cache" / "pdb_structures"
OUT_PATH = Path(__file__).resolve().parent / "l48_replication_out.json"

PDB_IDS = ["1UBQ", "1CRN", "1LYZ", "1MBN", "2LZM", "1PGA", "1TEN", "1SHG"]
MAX_SEQ_LEN = 300  # keep forward passes cheap; all 8 structures are well under this


@torch.no_grad()
def get_all_layer_head_attentions(model, tokenizer, sequence: str, device: str):
    """Returns attentions as a [n_layers, n_heads, seq_len, seq_len] numpy
    array (special tokens stripped), for one real sequence."""
    spaced_seq = " ".join(sequence[:MAX_SEQ_LEN])
    enc = tokenizer(spaced_seq, return_tensors="pt").to(device)
    out = model(**enc, output_attentions=True)
    # out.attentions: tuple of n_layers tensors, each [1, n_heads, seq_len, seq_len]
    stacked = torch.stack(out.attentions, dim=0).squeeze(1)  # [n_layers, n_heads, seq_len, seq_len]
    # strip special tokens (CLS at position 0, SEP at position -1 for BERT)
    stacked = stacked[:, :, 1:-1, 1:-1]
    return stacked.cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertModel.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers, {n_heads} heads", flush=True)

    # Accumulate per-(layer,head) attention-on-contacts fraction and
    # background rate ACROSS all structures (weighted by each structure's
    # own eligible-pair count), matching Vig et al.'s pooled-dataset metric
    # rather than averaging per-structure ratios (which would over-weight
    # small proteins).
    total_attention_on_contacts = np.zeros((n_layers, n_heads))
    total_eligible_attention = np.zeros((n_layers, n_heads))
    total_contacts = 0
    total_eligible_pairs = 0

    per_structure_info = []

    for pdb_id in PDB_IDS:
        pdb_path = PDB_DIR / f"{pdb_id}.pdb"
        sequence, contact_map = extract_sequence_and_contact_map(pdb_path)
        sequence = sequence[:MAX_SEQ_LEN]
        contact_map = contact_map[:MAX_SEQ_LEN, :MAX_SEQ_LEN]
        print(f"{pdb_id}: {len(sequence)} residues", flush=True)

        attentions = get_all_layer_head_attentions(model, tokenizer, sequence, device)
        # attentions shape [n_layers, n_heads, seq_len, seq_len] should match contact_map
        seq_len = contact_map.shape[0]
        if attentions.shape[-1] != seq_len:
            raise ValueError(f"{pdb_id}: attention seq_len {attentions.shape[-1]} != contact_map seq_len {seq_len}")

        ii, jj = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing="ij")
        from src.l38.l48_vig_contact_heads import MIN_SEQUENCE_SEPARATION
        eligible = np.abs(ii - jj) >= MIN_SEQUENCE_SEPARATION

        for layer in range(n_layers):
            for head in range(n_heads):
                attn = attentions[layer, head]
                total_attention_on_contacts[layer, head] += attn[eligible & contact_map].sum()
                total_eligible_attention[layer, head] += attn[eligible].sum()

        total_contacts += int((contact_map & eligible).sum())
        total_eligible_pairs += int(eligible.sum())
        per_structure_info.append({"pdb_id": pdb_id, "n_residues": seq_len, "n_contacts": int((contact_map & eligible).sum())})

    background_rate = total_contacts / total_eligible_pairs
    print(f"\npooled background contact rate: {background_rate:.4f}", flush=True)

    fraction = np.divide(total_attention_on_contacts, total_eligible_attention,
                          out=np.zeros_like(total_attention_on_contacts), where=total_eligible_attention > 0)
    enrichment = fraction / background_rate

    flat_idx = np.argsort(-enrichment.flatten())[:15]
    top_heads = [(int(i // n_heads), int(i % n_heads)) for i in flat_idx]

    print(f"\n=== TOP 15 MOST CONTACT-ENRICHED HEADS (pooled across {len(PDB_IDS)} real structures) ===", flush=True)
    results = {"background_rate": background_rate, "per_structure": per_structure_info, "top_heads": []}
    for layer, head in top_heads:
        entry = {
            "layer": layer, "head": head,
            "attention_on_contacts_fraction": float(fraction[layer, head]),
            "enrichment_ratio": float(enrichment[layer, head]),
        }
        results["top_heads"].append(entry)
        print(f"layer {layer:2d} head {head:2d}: fraction={entry['attention_on_contacts_fraction']:.4f} "
              f"enrichment={entry['enrichment_ratio']:.2f}x background", flush=True)

    results["full_fraction_matrix"] = fraction.tolist()
    results["full_enrichment_matrix"] = enrichment.tolist()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
