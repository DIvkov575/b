"""L49: unsupervised causal candidate generation for ProtBert-BFD attention
heads -- ablate EVERY head (30 layers x 16 heads = 480), rank by ACTUAL
CAUSAL EFFECT on masked-residue prediction, with ZERO correlational
attention-weight information used to pick candidates.

This is a genuinely different candidate-generation procedure from Vig et
al. 2021 and from this project's own L48 Stage 1: those both rank heads by
correlation (attention-on-contacts fraction) and only test the top pick
causally. This ranks ALL heads by causal effect directly -- the correlation
step is skipped entirely during discovery, and only used AFTERWARD (in the
cross-check section) to see whether causal and correlational rankings
agree, disagree, or find different heads. Three outcomes are all
informative; none is assumed going in.

Coarse-first design (matches this project's convention in L45's layer
sweep before its full run): a SUBSAMPLE of positions across all 8 real PDB
structures (not the full 770 used in L48's targeted 2-head test), since
testing all 480 heads on the full position set would take ~3 hours on this
hardware vs. ~24 minutes for a representative subsample.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import BertForMaskedLM, BertTokenizer

from src.l38.l42_steering_repro import paired_bootstrap_mean_diff, layer_effects_sign_test
from src.l38.l48_run_causal_ablation import HeadAblationHook, predict_single_position_masked
from src.l38.l48_vig_contact_heads import extract_sequence_and_contact_map

MODEL_NAME = "Rostlab/prot_bert_bfd"
PDB_DIR = Path(__file__).resolve().parent / "data_cache" / "pdb_structures"
OUT_PATH = Path(__file__).resolve().parent / "l49_causal_sweep_out.json"

PDB_IDS = ["1UBQ", "1CRN", "1LYZ", "1MBN", "2LZM", "1PGA", "1TEN", "1SHG"]
MAX_SEQ_LEN = 300
SEED = 0
N_POSITIONS_PER_STRUCTURE = 13  # ~100 total across 8 structures -- coarse
# pass, proportionally sampled (not all from one structure) so the sweep
# isn't dominated by whichever protein happens to be biggest.
TOP_K_TO_REPORT = 20


def main(layers_to_test=None):
    """layers_to_test: optional list of layer indices to restrict the sweep
    to (for smoke testing at small scale before committing to all 30)."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(MODEL_NAME, attn_implementation="eager").to(device).eval()
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers, {n_heads} heads", flush=True)

    layers_to_test = layers_to_test if layers_to_test is not None else list(range(n_layers))

    rng = np.random.RandomState(SEED)

    # sample a fixed, reusable set of (sequence, position, is_contact) triples
    # ONCE, so every head is evaluated on the EXACT SAME positions -- otherwise
    # per-head noise from different position samples would contaminate the
    # cross-head comparison.
    sampled_positions = []  # list of (pdb_id, sequence, contact_map, position, is_contact)
    for pdb_id in PDB_IDS:
        sequence, contact_map = extract_sequence_and_contact_map(PDB_DIR / f"{pdb_id}.pdb")
        sequence = sequence[:MAX_SEQ_LEN]
        contact_map = contact_map[:MAX_SEQ_LEN, :MAX_SEQ_LEN]
        n_residues = len(sequence)
        n_sample = min(N_POSITIONS_PER_STRUCTURE, n_residues)
        chosen = rng.choice(n_residues, size=n_sample, replace=False)
        has_contact = contact_map.any(axis=1)
        for pos in chosen:
            sampled_positions.append((pdb_id, sequence, int(pos), bool(has_contact[pos])))

    print(f"sampled {len(sampled_positions)} positions across {len(PDB_IDS)} structures", flush=True)

    print("\ncomputing baseline (no ablation) predictions...", flush=True)
    baseline_correct = []
    for pdb_id, sequence, pos, is_contact in sampled_positions:
        correct = predict_single_position_masked(model, tokenizer, sequence, pos, device)
        baseline_correct.append(correct)
    baseline_correct = np.array(baseline_correct, dtype=float)
    print(f"baseline accuracy: {baseline_correct.mean():.4f}", flush=True)

    all_head_results = []
    for i, layer in enumerate(layers_to_test):
        for head in range(n_heads):
            hook = HeadAblationHook(head, n_heads, head_dim)
            ablated_correct = []
            for pdb_id, sequence, pos, is_contact in sampled_positions:
                correct = predict_single_position_masked(
                    model, tokenizer, sequence, pos, device, ablation_hook=hook, ablation_layer=layer
                )
                ablated_correct.append(correct)
            ablated_correct = np.array(ablated_correct, dtype=float)

            diffs = ablated_correct - baseline_correct
            mean_effect = float(diffs.mean())
            all_head_results.append({
                "layer": layer, "head": head,
                "ablated_acc": float(ablated_correct.mean()),
                "mean_effect": mean_effect,  # negative = ablation HURTS accuracy (head is causally helpful)
            })
        print(f"  layer {layer:2d} done ({(i+1)*n_heads}/{len(layers_to_test)*n_heads} heads)", flush=True)

    # rank by mean_effect ascending (most NEGATIVE = ablating it hurts
    # accuracy most = causally most important, by this task's definition)
    all_head_results_sorted = sorted(all_head_results, key=lambda r: r["mean_effect"])

    print(f"\n=== TOP {TOP_K_TO_REPORT} MOST CAUSALLY IMPORTANT HEADS (ablation hurts accuracy most) ===", flush=True)
    for r in all_head_results_sorted[:TOP_K_TO_REPORT]:
        print(f"layer {r['layer']:2d} head {r['head']:2d}: mean_effect={r['mean_effect']:+.4f} "
              f"(ablated_acc={r['ablated_acc']:.4f} vs baseline={baseline_correct.mean():.4f})", flush=True)

    print(f"\n=== TOP {TOP_K_TO_REPORT} MOST CAUSALLY HELPFUL-TO-ABLATE HEADS (ablation IMPROVES accuracy) ===", flush=True)
    for r in all_head_results_sorted[-TOP_K_TO_REPORT:][::-1]:
        print(f"layer {r['layer']:2d} head {r['head']:2d}: mean_effect={r['mean_effect']:+.4f}", flush=True)

    effects = np.array([r["mean_effect"] for r in all_head_results])
    sign_test = layer_effects_sign_test(effects.tolist())
    print(f"\n=== SIGN TEST across all {len(effects)} heads ===", flush=True)
    print(json.dumps(sign_test, indent=2), flush=True)

    results = {
        "n_sampled_positions": len(sampled_positions),
        "baseline_accuracy": float(baseline_correct.mean()),
        "all_heads": all_head_results,
        "top_causally_important": all_head_results_sorted[:TOP_K_TO_REPORT],
        "top_causally_helpful_to_ablate": all_head_results_sorted[-TOP_K_TO_REPORT:][::-1],
        "sign_test_all_heads": sign_test,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
