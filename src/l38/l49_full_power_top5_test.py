"""L49 follow-up: L49's 480-head causal sweep was a coarse, low-power pass
(104 positions, no per-head paired-bootstrap significance test) -- its own
"What this does NOT show" section flags exactly this as the natural next
step. This reruns L48's full 770-position, proper paired-bootstrap
significance test (identical method to L48 Stage 2) on the top-5 heads by
magnitude that L49's coarse sweep flagged:
  (12, 11), (14, 15), (15, 8), (18, 3), (24, 9)
-- all five tied at mean_effect=-0.0385 in the coarse pass, the single
largest ablation-hurts-accuracy magnitude found across all 480 heads.

Confirms/disconfirms whether the coarse ranking's TOP PICKS are real signal
or noise at n=104, by testing each individually against baseline on the
full 770-position set with a real confidence interval -- not just trusting
the coarse point estimate's rank order.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from src.l38.l42_steering_repro import paired_bootstrap_mean_diff
from src.l38.l48_run_causal_ablation import HeadAblationHook, predict_single_position_masked
from src.l38.l48_vig_contact_heads import extract_sequence_and_contact_map

MODEL_NAME = "Rostlab/prot_bert_bfd"
PDB_DIR = Path(__file__).resolve().parent / "data_cache" / "pdb_structures"
OUT_PATH = Path(__file__).resolve().parent / "l49_full_power_top5_out.json"

PDB_IDS = ["1UBQ", "1CRN", "1LYZ", "1MBN", "2LZM", "1PGA", "1TEN", "1SHG"]
MAX_SEQ_LEN = 300

# top-5 by magnitude from L49's coarse sweep (src/l38/l49_causal_sweep_out.json,
# "top_causally_important"), all tied at mean_effect=-0.0385 at n=104.
TOP5_HEADS = [(12, 11), (14, 15), (15, 8), (18, 3), (24, 9)]


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"model loaded: {MODEL_NAME}, {n_heads} heads, head_dim={head_dim}", flush=True)

    # collect ALL residues (contact + non-contact, full 770) across the same
    # 8 structures L48/L49 used, matching L48 Stage 2's full-position scope
    # (not L49's 104-position coarse subsample).
    all_positions = []  # (sequence, position)
    for pdb_id in PDB_IDS:
        sequence, contact_map = extract_sequence_and_contact_map(PDB_DIR / f"{pdb_id}.pdb")
        sequence = sequence[:MAX_SEQ_LEN]
        n_residues = len(sequence)
        for pos in range(n_residues):
            all_positions.append((pdb_id, sequence, pos))
    print(f"total positions across {len(PDB_IDS)} structures: {len(all_positions)}", flush=True)

    print("\ncomputing baseline (no ablation) predictions...", flush=True)
    baseline_correct = []
    for pdb_id, sequence, pos in all_positions:
        baseline_correct.append(predict_single_position_masked(model, tokenizer, sequence, pos, device))
    baseline_correct = np.array(baseline_correct, dtype=float)
    print(f"baseline accuracy: {baseline_correct.mean():.4f} (n={len(baseline_correct)})", flush=True)

    results = {"n_positions": len(all_positions), "baseline_accuracy": float(baseline_correct.mean()), "per_head": {}}

    for layer, head in TOP5_HEADS:
        hook = HeadAblationHook(head, n_heads, head_dim)
        ablated_correct = []
        for pdb_id, sequence, pos in all_positions:
            ablated_correct.append(
                predict_single_position_masked(model, tokenizer, sequence, pos, device, ablation_hook=hook, ablation_layer=layer)
            )
        ablated_correct = np.array(ablated_correct, dtype=float)

        bootstrap = paired_bootstrap_mean_diff(baseline_correct, ablated_correct, n_boot=10000, seed=0)
        results["per_head"][f"{layer}_{head}"] = {
            "layer": layer, "head": head,
            "ablated_accuracy": float(ablated_correct.mean()),
            "coarse_sweep_mean_effect_at_n104": -0.038461538461538464,
            **bootstrap,
        }
        print(f"layer {layer:2d} head {head:2d}: ablated_acc={ablated_correct.mean():.4f} "
              f"diff={bootstrap['point_estimate']:+.4f} [{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] "
              f"sig={bootstrap['significant_at_95pct']} (n={bootstrap['n']})", flush=True)

    n_significant = sum(1 for e in results["per_head"].values() if e["significant_at_95pct"])
    results["n_significant_of_5"] = n_significant
    print(f"\n=== {n_significant}/5 heads significant at full power (n={len(all_positions)}) ===", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
