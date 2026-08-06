"""L48 follow-up: does ablating the TOP-30 contact-enriched heads
SIMULTANEOUSLY hurt prediction, even though ablating any ONE of them alone
(L48 Stage 2, docs/L48_VIG_CAUSAL_TEST.md) did not?

L48 tested exactly one head (Vig's #1 pick) plus one low-enrichment control,
one at a time, and found no significant effect on either. That result
cannot distinguish two explanations: (a) attention correlation with contacts
genuinely doesn't reflect causal importance here, or (b) the network is
REDUNDANT -- many heads carry overlapping structural signal, so losing any
single one is invisible but losing many at once would not be. This tests
(b) directly, reusing L48's exact task, structures, and significance test.

Design: three conditions on the SAME 770 real residues from the same 8 PDB
structures used throughout L48/L49:
  1. baseline (no ablation)
  2. top-30 contact-enriched heads ablated simultaneously (by attention
     enrichment ranking, l48_replication_out.json)
  3. a SIZE-MATCHED random-head control: 30 heads chosen uniformly at random
     (fixed seed) from the remaining 450, ablated simultaneously -- controls
     for "ablating any 30 heads out of 480 just generically degrades the
     model," which the single-head test couldn't rule out either.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from src.l38.l42_steering_repro import paired_bootstrap_mean_diff
from src.l38.l48_vig_contact_heads import extract_sequence_and_contact_map

MODEL_NAME = "Rostlab/prot_bert_bfd"
PDB_DIR = Path(__file__).resolve().parent / "data_cache" / "pdb_structures"
REPLICATION_PATH = Path(__file__).resolve().parent / "l48_replication_out.json"
OUT_PATH = Path(__file__).resolve().parent / "l48_multihead_redundancy_out.json"

PDB_IDS = ["1UBQ", "1CRN", "1LYZ", "1MBN", "2LZM", "1PGA", "1TEN", "1SHG"]
MAX_SEQ_LEN = 300
N_TOP_HEADS = 30  # matches L49's design convention (round number, not cherry-picked to a result)
SEED = 0


class MultiHeadAblationHook:
    """Zeros out MULTIPLE specified heads' contributions within one layer's
    merged attention output -- same reshape-merge mechanism as L48's
    single-head HeadAblationHook, generalized to a set of head indices."""

    def __init__(self, heads: list, num_heads: int, head_dim: int):
        self.heads = heads
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        current = output[0] if is_tuple else output
        per_head = current.view(*current.shape[:-1], self.num_heads, self.head_dim).clone()
        for h in self.heads:
            per_head[..., h, :] = 0.0
        ablated = per_head.view(*current.shape)
        if is_tuple:
            return (ablated,) + output[1:]
        return ablated


def predict_single_position_masked(model, tokenizer, sequence, position, device, ablation_hooks_by_layer=None):
    """ablation_hooks_by_layer: dict layer_idx -> hook, registered on ALL
    given layers simultaneously (unlike L48's single-layer version)."""
    spaced_seq = " ".join(sequence)
    enc = tokenizer(spaced_seq, return_tensors="pt").to(device)
    input_ids = enc["input_ids"].clone()
    token_pos = position + 1
    true_id = input_ids[0, token_pos].item()
    input_ids[0, token_pos] = tokenizer.mask_token_id

    handles = []
    if ablation_hooks_by_layer:
        for layer, hook in ablation_hooks_by_layer.items():
            handles.append(model.bert.encoder.layer[layer].attention.self.register_forward_hook(hook))

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=enc["attention_mask"])
    finally:
        for h in handles:
            h.remove()

    predicted_id = out.logits[0, token_pos].argmax().item()
    return predicted_id == true_id


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    with open(REPLICATION_PATH) as f:
        replication = json.load(f)
    # `top_heads` in the replication output only stores the top 15; derive
    # the full top-30 ranking directly from the full 30x16 enrichment matrix
    # instead (same underlying Stage-1 data, just not truncated).
    enrichment_matrix = np.array(replication["full_enrichment_matrix"])  # [n_layers, n_heads]
    n_layers_repl, n_heads_repl = enrichment_matrix.shape
    all_pairs_ranked = sorted(
        ((l, h, enrichment_matrix[l, h]) for l in range(n_layers_repl) for h in range(n_heads_repl)),
        key=lambda x: -x[2],
    )
    top_30 = [(l, h) for l, h, _ in all_pairs_ranked[:N_TOP_HEADS]]
    print(f"top-{N_TOP_HEADS} contact-enriched heads (by enrichment ratio, {all_pairs_ranked[0][2]:.2f}x "
          f"down to {all_pairs_ranked[N_TOP_HEADS-1][2]:.2f}x): {top_30}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers, {n_heads} heads/layer", flush=True)

    all_head_pairs = [(l, h) for l in range(n_layers) for h in range(n_heads)]
    top_set = set(top_30)
    remaining = [p for p in all_head_pairs if p not in top_set]
    rng = np.random.RandomState(SEED)
    random_30_idx = rng.choice(len(remaining), size=N_TOP_HEADS, replace=False)
    random_30 = [remaining[i] for i in random_30_idx]
    print(f"random-control {N_TOP_HEADS} heads (uniform from remaining {len(remaining)}): {random_30}", flush=True)

    def hooks_for(head_pairs):
        by_layer = {}
        for layer, head in head_pairs:
            by_layer.setdefault(layer, []).append(head)
        return {layer: MultiHeadAblationHook(heads, n_heads, head_dim) for layer, heads in by_layer.items()}

    top_hooks = hooks_for(top_30)
    random_hooks = hooks_for(random_30)

    results = {
        "n_top_heads": N_TOP_HEADS, "top_30_heads": top_30, "random_30_heads": random_30,
        "per_structure": [],
    }

    total_n = {"contact": 0, "non_contact": 0}
    per_position_records = {"contact": [], "non_contact": []}

    for pdb_id in PDB_IDS:
        sequence, contact_map = extract_sequence_and_contact_map(PDB_DIR / f"{pdb_id}.pdb")
        sequence = sequence[:MAX_SEQ_LEN]
        contact_map = contact_map[:MAX_SEQ_LEN, :MAX_SEQ_LEN]
        has_contact = contact_map.any(axis=1)
        print(f"\n{pdb_id}: {len(sequence)} residues, {has_contact.sum()} contact-bearing", flush=True)

        struct_result = {"pdb_id": pdb_id, "n_residues": len(sequence), "n_contact_bearing": int(has_contact.sum())}

        for category, positions in [("contact", np.where(has_contact)[0]), ("non_contact", np.where(~has_contact)[0])]:
            baseline_correct = 0
            top_ablated_correct = 0
            random_ablated_correct = 0
            for pos in positions:
                pos = int(pos)
                b = predict_single_position_masked(model, tokenizer, sequence, pos, device)
                t = predict_single_position_masked(model, tokenizer, sequence, pos, device, ablation_hooks_by_layer=top_hooks)
                r = predict_single_position_masked(model, tokenizer, sequence, pos, device, ablation_hooks_by_layer=random_hooks)
                baseline_correct += b
                top_ablated_correct += t
                random_ablated_correct += r
                per_position_records[category].append({"baseline": int(b), "top30_ablated": int(t), "random30_ablated": int(r)})
            n = len(positions)
            struct_result[f"{category}_n"] = n
            struct_result[f"{category}_baseline_acc"] = baseline_correct / n if n > 0 else None
            struct_result[f"{category}_top30_ablated_acc"] = top_ablated_correct / n if n > 0 else None
            struct_result[f"{category}_random30_ablated_acc"] = random_ablated_correct / n if n > 0 else None
            total_n[category] += n
            print(f"  {category} (n={n}): baseline={struct_result[f'{category}_baseline_acc']:.3f} "
                  f"top30_ablated={struct_result[f'{category}_top30_ablated_acc']:.3f} "
                  f"random30_ablated={struct_result[f'{category}_random30_ablated_acc']:.3f}", flush=True)

        results["per_structure"].append(struct_result)

    print("\n=== POOLED ACROSS ALL STRUCTURES (with paired-bootstrap significance) ===", flush=True)
    pooled = {}
    for category in ["contact", "non_contact"]:
        n = total_n[category]
        records = per_position_records[category]
        baseline_arr = np.array([r["baseline"] for r in records], dtype=float)
        top_arr = np.array([r["top30_ablated"] for r in records], dtype=float)
        random_arr = np.array([r["random30_ablated"] for r in records], dtype=float)

        top_bootstrap = paired_bootstrap_mean_diff(baseline_arr, top_arr, n_boot=10000, seed=0)
        random_bootstrap = paired_bootstrap_mean_diff(baseline_arr, random_arr, n_boot=10000, seed=0)
        top_vs_random_bootstrap = paired_bootstrap_mean_diff(random_arr, top_arr, n_boot=10000, seed=0)

        pooled[category] = {
            "n": n,
            "baseline_acc": baseline_arr.mean(),
            "top30_ablated_acc": top_arr.mean(),
            "random30_ablated_acc": random_arr.mean(),
            "top30_ablation_vs_baseline": top_bootstrap,
            "random30_ablation_vs_baseline": random_bootstrap,
            "top30_vs_random30_ablation": top_vs_random_bootstrap,
        }
        print(f"{category} (n={n}): baseline={pooled[category]['baseline_acc']:.4f}", flush=True)
        print(f"  top30_ablated vs baseline: diff={top_bootstrap['point_estimate']:+.4f} "
              f"[{top_bootstrap['ci_lower']:.4f}, {top_bootstrap['ci_upper']:.4f}] sig={top_bootstrap['significant_at_95pct']}", flush=True)
        print(f"  random30_ablated vs baseline: diff={random_bootstrap['point_estimate']:+.4f} "
              f"[{random_bootstrap['ci_lower']:.4f}, {random_bootstrap['ci_upper']:.4f}] sig={random_bootstrap['significant_at_95pct']}", flush=True)
        print(f"  top30 vs random30 (does top set matter MORE): diff={top_vs_random_bootstrap['point_estimate']:+.4f} "
              f"[{top_vs_random_bootstrap['ci_lower']:.4f}, {top_vs_random_bootstrap['ci_upper']:.4f}] sig={top_vs_random_bootstrap['significant_at_95pct']}", flush=True)

    results["pooled"] = pooled
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
