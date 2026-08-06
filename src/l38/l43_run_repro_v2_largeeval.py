"""L43 follow-up: does the weak, non-significant low-alpha solubility hint
(alpha=0.25: 63% of sequences move positive, mean diff +0.014, not
significant at n=60 -- docs/L43_SOLUBILITY_STEERING.md) resolve with a
larger held-out eval set? Reuses L43's exact vectors, hooks, scorer, and
degeneracy filter unchanged (imports from l43_run_repro.py) -- only
N_EVAL_SEQS increases (60 -> 300) and the alpha grid is restricted to
[0.1, 0.25, 0.5], since 1.0/2.0 were already established as beyond the
trustworthy (non-degenerate) range for this model/generation setup.

This is a power check, not a new method: if the same small, positive-leaning
effect survives a 5x larger sample and clears significance, that's a real
(if still small) generalization of L42's harness to solubility. If it stays
sub-significant or reverses, the original AMBIGUOUS verdict stands and the
harness genuinely doesn't generalize to solubility with this recipe.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
)
from src.l38.l43_run_repro import (
    MODEL_NAME,
    TRAIN_PATH,
    MAX_SEQ_LEN,
    N_VECTOR_SEQS_PER_GROUP,
    MASK_FRACTION,
    SEED,
    MultiLayerSteeringHook,
    mean_pooled_activation_all_layers,
    mask_fill_generate,
    score_solubility_proxy,
)
from src.l38.l43_solubility_steering import solubility_proxy_excluding

OUT_PATH = Path(__file__).resolve().parent / "l43_repro_out" / "results_large_eval.json"

N_EVAL_SEQS = 300  # 5x L43's original n=60
ALPHAS = [0.1, 0.25, 0.5]  # restricted to the non-degenerate range established in L43
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(TRAIN_PATH)
    df = df[df["sequences"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]

    insoluble_pool = vector_pool[vector_pool["labels"] == 0]["sequences"].tolist()
    soluble_pool = vector_pool[vector_pool["labels"] == 1]["sequences"].tolist()
    low_group = insoluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    high_group = soluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} insoluble, {len(high_group)} soluble", flush=True)

    eval_insoluble = eval_pool[eval_pool["labels"] == 0]["sequences"].tolist()
    eval_sequences = eval_insoluble[:N_EVAL_SEQS]
    print(f"eval sequences (insoluble, held out, disjoint from vector-building): {len(eval_sequences)}", flush=True)
    assert len(eval_sequences) == N_EVAL_SEQS, f"only {len(eval_sequences)} eval sequences available, need {N_EVAL_SEQS}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding insoluble group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding soluble group (all layers)...", flush=True)
    high_activations = mean_pooled_activation_all_layers(model, tokenizer, high_group, device)

    steering_vectors = {}
    for layer in range(n_layers):
        vec = difference_of_means_vector(low_activations[layer], high_activations[layer])
        steering_vectors[layer] = torch.tensor(vec, dtype=torch.float32, device=device)

    rng2 = np.random.RandomState(SEED + 1)
    random_vectors = {
        layer: torch.tensor(rng2.normal(size=vec.shape[0]), dtype=torch.float32, device=device)
        for layer, vec in steering_vectors.items()
    }
    for layer in random_vectors:
        real_norm = steering_vectors[layer].norm()
        random_vectors[layer] = random_vectors[layer] / random_vectors[layer].norm() * real_norm

    def apply_hooks(vectors, alpha):
        handles = []
        for layer, vec in vectors.items():
            hook = MultiLayerSteeringHook(vec, alpha)
            handles.append(model.esm.encoder.layer[layer].register_forward_hook(hook))
        return handles

    def remove_hooks(handles):
        for h in handles:
            h.remove()

    def generate_then_score(vectors, alpha):
        generated = []
        handles = apply_hooks(vectors, alpha) if alpha != 0.0 else []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            remove_hooks(handles)
        scores = score_solubility_proxy(generated)
        return generated, scores

    results = {"n_eval_seqs": N_EVAL_SEQS, "alphas": ALPHAS, "real_direction": {}, "random_control": {}}

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)
    results["baseline"] = {
        "mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()), "n": len(baseline_scores),
        "n_degenerate": int(baseline_degenerate.sum()),
    }

    real_by_alpha = {}
    random_by_alpha = {}
    all_real_generated = {}

    for alpha in ALPHAS:
        print(f"\n=== real_direction, alpha={alpha} ===", flush=True)
        generated, scores = generate_then_score(steering_vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        real_by_alpha[alpha] = (generated, scores, degenerate)
        all_real_generated[alpha] = generated
        print(f"mean score: {scores.mean():.4f}, degenerate: {degenerate.sum()}/{len(degenerate)}", flush=True)

        print(f"=== random_control, alpha={alpha} ===", flush=True)
        generated, scores = generate_then_score(random_vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        random_by_alpha[alpha] = (generated, scores, degenerate)
        print(f"mean score: {scores.mean():.4f}, degenerate: {degenerate.sum()}/{len(degenerate)}", flush=True)

    real_vs_random_by_alpha = {}
    for alpha in ALPHAS:
        real_generated, real_scores, real_degenerate = real_by_alpha[alpha]
        random_generated, random_scores, random_degenerate = random_by_alpha[alpha]
        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())
        results["real_direction"][alpha] = {"mean": float(real_scores.mean()), "n_degenerate": int(real_degenerate.sum())}
        results["random_control"][alpha] = {"mean": float(random_scores.mean()), "n_degenerate": int(random_degenerate.sum())}
        if n_kept < MIN_NONDEGENERATE_PAIRS:
            real_vs_random_by_alpha[alpha] = {
                "point_estimate": None, "ci_lower": None, "ci_upper": None,
                "significant_at_95pct": False, "n": n_kept,
                "excluded_reason": f"only {n_kept} non-degenerate pairs",
            }
            continue
        bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
        real_vs_random_by_alpha[alpha] = bootstrap
        pct_positive = float((real_scores[keep] > random_scores[keep]).mean())
        real_vs_random_by_alpha[alpha]["pct_sequences_real_beats_random"] = pct_positive
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={pct_positive:.3f}", flush=True)

    # residue-exclusion robustness check on whichever alpha shows the
    # strongest effect, mirroring L42/L43's leucine/A-G exclusion discipline --
    # find the dominant substituted residue(s) in the real-direction condition
    # relative to baseline and rerun the score with them excluded.
    best_alpha = max(
        (a for a in ALPHAS if real_vs_random_by_alpha[a].get("significant_at_95pct")),
        key=lambda a: real_vs_random_by_alpha[a]["point_estimate"],
        default=None,
    )
    robustness_check = None
    if best_alpha is not None:
        from collections import Counter
        real_generated, _, real_degenerate = real_by_alpha[best_alpha]
        counts = Counter()
        for seq, base in zip(real_generated, baseline_generated):
            for a, b in zip(seq, base):
                if a != b:
                    counts[a] += 1
        top_residues = frozenset(r for r, _ in counts.most_common(2))
        print(f"\ndominant substituted residues at alpha={best_alpha}: {counts.most_common(5)}", flush=True)

        keep = ~real_degenerate & ~random_by_alpha[best_alpha][2] & ~baseline_degenerate
        real_excl_scores = np.array([
            solubility_proxy_excluding(s, top_residues) for s in np.array(real_by_alpha[best_alpha][0])[keep]
        ])
        random_excl_scores = np.array([
            solubility_proxy_excluding(s, top_residues) for s in np.array(random_by_alpha[best_alpha][0])[keep]
        ])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl_scores, real_excl_scores, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha, "excluded_residues": sorted(top_residues),
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): diff={excl_bootstrap['point_estimate']:.4f} "
              f"[{excl_bootstrap['ci_lower']:.4f}, {excl_bootstrap['ci_upper']:.4f}] sig={excl_bootstrap['significant_at_95pct']}", flush=True)

    verdict = {
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "robustness_check": robustness_check,
        "decision": (
            "PASS_ROBUST" if best_alpha is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
            else "PASS_ARTIFACT" if best_alpha is not None
            else "STILL_INCONCLUSIVE"
        ),
    }

    print("\n=== L43-large-eval VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results["verdict"] = verdict
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
