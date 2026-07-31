"""L48 Task A, Stage 2: the actual causal test. Vig et al. found layer/head
attention that strongly aligns with real contacts (replicated in Stage 1:
layer 5 head 13 on ProtBert-BFD, 12.9x enrichment over background on real
PDB structures) but never tested whether that head's output CAUSALLY
matters for any downstream task -- their own words: "purely associative."

Task: single-position masked-residue prediction, one contact-bearing
residue at a time (all OTHER positions stay visible, including the distant
residues in real 3D contact -- so a model that "uses" structural context
non-locally should do BETTER than one that only uses local sequence
patterns). Measure accuracy under three conditions per real structure:
  1. baseline (no ablation)
  2. top-contact-head ablated (layer 5, head 13 zeroed)
  3. control-head ablated (layer 17, head 1 -- the LOWEST-enrichment real
     head found in Stage 1, 0.055x background -- a genuine "this head
     barely touches contacts at all" control, not just a random pick)

If ablating the top contact head hurts accuracy specifically at
contact-bearing positions MORE than ablating the control head, that's the
first real causal evidence for Vig et al.'s 5-year-old correlational
finding. If it doesn't, that's equally real: correlation without causal
necessity, matching this project's L41/L45 experience that
attention/activation alignment doesn't guarantee causal importance.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertModel, BertTokenizer

from src.l38.l48_vig_contact_heads import extract_sequence_and_contact_map

MODEL_NAME = "Rostlab/prot_bert_bfd"
PDB_DIR = Path(__file__).resolve().parent / "data_cache" / "pdb_structures"
OUT_PATH = Path(__file__).resolve().parent / "l48_causal_ablation_out.json"

PDB_IDS = ["1UBQ", "1CRN", "1LYZ", "1MBN", "2LZM", "1PGA", "1TEN", "1SHG"]
MAX_SEQ_LEN = 300

TOP_CONTACT_HEAD = (5, 13)  # from Stage 1 replication: 12.9x enrichment
CONTROL_HEAD = (17, 1)  # from Stage 1 replication: 0.055x enrichment (lowest of all 480)


class HeadAblationHook:
    """Zeros out exactly one attention head's contribution to a layer's
    merged output -- same head-slicing mechanism verified in Phase 0
    feasibility checks (l47_activation_patching.py), applied here to
    BertSelfAttention instead of ESM2's EsmSelfAttention (same underlying
    reshape-merge pattern, confirmed directly before use).
    """

    def __init__(self, head: int, num_heads: int, head_dim: int):
        self.head = head
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        current = output[0] if is_tuple else output
        per_head = current.view(*current.shape[:-1], self.num_heads, self.head_dim).clone()
        per_head[..., self.head, :] = 0.0
        ablated = per_head.view(*current.shape)
        if is_tuple:
            return (ablated,) + output[1:]
        return ablated


@torch.no_grad()
def predict_single_position_masked(model, tokenizer, sequence: str, position: int, device: str, ablation_hook=None, ablation_layer=None):
    """Mask exactly ONE position (all others stay as real residues),
    predict it, return whether the prediction matches the true residue."""
    spaced_seq = " ".join(sequence)
    enc = tokenizer(spaced_seq, return_tensors="pt").to(device)
    input_ids = enc["input_ids"].clone()

    # position i in `sequence` maps to token index i+1 (CLS at index 0)
    token_pos = position + 1
    true_id = input_ids[0, token_pos].item()
    input_ids[0, token_pos] = tokenizer.mask_token_id

    handle = None
    if ablation_hook is not None:
        handle = model.bert.encoder.layer[ablation_layer].attention.self.register_forward_hook(ablation_hook)

    try:
        out = model(input_ids=input_ids, attention_mask=enc["attention_mask"])
    finally:
        if handle is not None:
            handle.remove()

    predicted_id = out.logits[0, token_pos].argmax().item()
    return predicted_id == true_id


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    from transformers import BertForMaskedLM
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"model loaded: {MODEL_NAME}, {n_heads} heads, head_dim={head_dim}", flush=True)

    top_layer, top_head = TOP_CONTACT_HEAD
    ctrl_layer, ctrl_head = CONTROL_HEAD
    top_hook = HeadAblationHook(top_head, n_heads, head_dim)
    ctrl_hook = HeadAblationHook(ctrl_head, n_heads, head_dim)

    results = {"top_contact_head": TOP_CONTACT_HEAD, "control_head": CONTROL_HEAD, "per_structure": []}

    total_baseline_correct = {"contact": 0, "non_contact": 0}
    total_top_ablated_correct = {"contact": 0, "non_contact": 0}
    total_ctrl_ablated_correct = {"contact": 0, "non_contact": 0}
    total_n = {"contact": 0, "non_contact": 0}
    # per-position correctness (0/1), pooled across all structures, for a
    # real paired significance test -- aggregate accuracy alone isn't
    # enough to tell a real effect from noise (this project's established
    # discipline, see L42/L43/L45/L47's paired-bootstrap tests).
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
            ctrl_ablated_correct = 0
            for pos in positions:
                pos = int(pos)
                b = predict_single_position_masked(model, tokenizer, sequence, pos, device)
                t = predict_single_position_masked(
                    model, tokenizer, sequence, pos, device, ablation_hook=top_hook, ablation_layer=top_layer
                )
                c = predict_single_position_masked(
                    model, tokenizer, sequence, pos, device, ablation_hook=ctrl_hook, ablation_layer=ctrl_layer
                )
                baseline_correct += b
                top_ablated_correct += t
                ctrl_ablated_correct += c
                per_position_records[category].append({"baseline": int(b), "top_ablated": int(t), "ctrl_ablated": int(c)})
            n = len(positions)
            struct_result[f"{category}_n"] = n
            struct_result[f"{category}_baseline_acc"] = baseline_correct / n if n > 0 else None
            struct_result[f"{category}_top_ablated_acc"] = top_ablated_correct / n if n > 0 else None
            struct_result[f"{category}_ctrl_ablated_acc"] = ctrl_ablated_correct / n if n > 0 else None

            total_baseline_correct[category] += baseline_correct
            total_top_ablated_correct[category] += top_ablated_correct
            total_ctrl_ablated_correct[category] += ctrl_ablated_correct
            total_n[category] += n

            print(f"  {category} (n={n}): baseline={struct_result[f'{category}_baseline_acc']:.3f} "
                  f"top_ablated={struct_result[f'{category}_top_ablated_acc']:.3f} "
                  f"ctrl_ablated={struct_result[f'{category}_ctrl_ablated_acc']:.3f}", flush=True)

        results["per_structure"].append(struct_result)

    from src.l38.l42_steering_repro import paired_bootstrap_mean_diff

    print("\n=== POOLED ACROSS ALL STRUCTURES (with paired-bootstrap significance) ===", flush=True)
    pooled = {}
    for category in ["contact", "non_contact"]:
        n = total_n[category]
        records = per_position_records[category]
        baseline_arr = np.array([r["baseline"] for r in records], dtype=float)
        top_arr = np.array([r["top_ablated"] for r in records], dtype=float)
        ctrl_arr = np.array([r["ctrl_ablated"] for r in records], dtype=float)

        top_bootstrap = paired_bootstrap_mean_diff(baseline_arr, top_arr, n_boot=10000, seed=0)
        ctrl_bootstrap = paired_bootstrap_mean_diff(baseline_arr, ctrl_arr, n_boot=10000, seed=0)
        # is the TOP head's ablation effect different from the CONTROL head's
        # ablation effect? (the real question: does the contact-enriched
        # head matter MORE than an arbitrary/low-enrichment head)
        top_vs_ctrl_bootstrap = paired_bootstrap_mean_diff(ctrl_arr, top_arr, n_boot=10000, seed=0)

        pooled[category] = {
            "n": n,
            "baseline_acc": total_baseline_correct[category] / n,
            "top_ablated_acc": total_top_ablated_correct[category] / n,
            "ctrl_ablated_acc": total_ctrl_ablated_correct[category] / n,
            "top_ablation_vs_baseline": top_bootstrap,
            "ctrl_ablation_vs_baseline": ctrl_bootstrap,
            "top_vs_ctrl_ablation": top_vs_ctrl_bootstrap,
        }
        print(f"{category} (n={n}): baseline={pooled[category]['baseline_acc']:.4f}", flush=True)
        print(f"  top_ablated vs baseline: diff={top_bootstrap['point_estimate']:+.4f} "
              f"[{top_bootstrap['ci_lower']:.4f}, {top_bootstrap['ci_upper']:.4f}] sig={top_bootstrap['significant_at_95pct']}", flush=True)
        print(f"  ctrl_ablated vs baseline: diff={ctrl_bootstrap['point_estimate']:+.4f} "
              f"[{ctrl_bootstrap['ci_lower']:.4f}, {ctrl_bootstrap['ci_upper']:.4f}] sig={ctrl_bootstrap['significant_at_95pct']}", flush=True)
        print(f"  top vs ctrl ablation (does the contact head matter MORE): diff={top_vs_ctrl_bootstrap['point_estimate']:+.4f} "
              f"[{top_vs_ctrl_bootstrap['ci_lower']:.4f}, {top_vs_ctrl_bootstrap['ci_upper']:.4f}] sig={top_vs_ctrl_bootstrap['significant_at_95pct']}", flush=True)

    results["pooled"] = pooled

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
