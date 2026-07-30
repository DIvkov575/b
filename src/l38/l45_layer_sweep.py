"""L45: per-layer causal sufficiency sweep for L42's validated thermostability
steering vector.

L42 applies the steering vector to ALL 33 layers simultaneously and found a
real effect at alpha in [0.1, 0.5] (docs/L42_STEERING_REPRO.md). That
confirms the vector CAN steer, but not WHICH layers are doing the causal
work versus just going along for the ride. This sweeps alpha=0.25 (L42's
strongest clean, non-collapsing signal) across each layer INDIVIDUALLY --
same vectors, same eval sequences, same scorer, same degeneracy filter, same
paired-bootstrap test as L42 -- to find which specific layers are causally
sufficient to reproduce a real (vs. random-control) effect on their own.

This is activation-patching-style layer localization, not a new claim about
thermostability: a real effect at one layer means that layer's activation
change is, by itself, enough to shift the model's downstream behavior in the
validated direction. No effect at any single layer (while the all-layers
version works) would mean the effect is a genuinely distributed/emergent
property of the full residual stream, not attributable to any localized
computation -- itself a real, checkable finding either way.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_run_repro import (
    ALPHAS,
    MASK_FRACTION,
    MODEL_NAME,
    N_EVAL_SEQS,
    N_VECTOR_SEQS_PER_GROUP,
    DATA_PATH,
    MAX_SEQ_LEN,
    SEED,
    MultiLayerSteeringHook,
    mask_fill_generate,
    mean_pooled_activation_all_layers,
    score_thermostability_proxy,
)
from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
    split_by_percentile,
)

OUT_PATH = Path(__file__).resolve().parent / "l45_layer_sweep_out.json"
SWEEP_ALPHA = 0.25  # L42's cleanest non-collapsing signal (diff=+0.022, all layers)
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30  # same trust guard as L42 -- see docs/L42_STEERING_REPRO.md


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]

    low_group, high_group = split_by_percentile(
        vector_pool["sequence"].tolist(), vector_pool["label"].values, low_pct=20.0, high_pct=80.0
    )
    low_group = low_group[:N_VECTOR_SEQS_PER_GROUP]
    high_group = high_group[:N_VECTOR_SEQS_PER_GROUP]

    eval_pool_sorted = eval_pool.sort_values("label")
    eval_sequences = eval_pool_sorted["sequence"].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences: {len(eval_sequences)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-Tm group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-Tm group (all layers)...", flush=True)
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

    def generate_with_single_layer_hook(vector, layer, alpha):
        hook = MultiLayerSteeringHook(vector, alpha)
        handle = model.esm.encoder.layer[layer].register_forward_hook(hook)
        generated = []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            handle.remove()
        scores = score_thermostability_proxy(generated)
        return generated, scores

    print("\n=== baseline (no hooks) ===", flush=True)
    baseline_generated = [
        mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device)
        for i, seq in enumerate(eval_sequences)
    ]
    baseline_scores = score_thermostability_proxy(baseline_generated)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)

    results = {"sweep_alpha": SWEEP_ALPHA, "baseline_mean": float(baseline_scores.mean()), "by_layer": {}}

    for layer in range(n_layers):
        real_generated, real_scores = generate_with_single_layer_hook(steering_vectors[layer], layer, SWEEP_ALPHA)
        random_generated, random_scores = generate_with_single_layer_hook(random_vectors[layer], layer, SWEEP_ALPHA)

        real_degenerate = np.array([is_degenerate_sequence(s) for s in real_generated])
        random_degenerate = np.array([is_degenerate_sequence(s) for s in random_generated])
        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())

        if n_kept < MIN_NONDEGENERATE_PAIRS:
            entry = {
                "real_mean": float(real_scores.mean()), "random_mean": float(random_scores.mean()),
                "n_degenerate_real": int(real_degenerate.sum()), "n_degenerate_random": int(random_degenerate.sum()),
                "point_estimate": None, "ci_lower": None, "ci_upper": None, "significant_at_95pct": False,
                "n": n_kept, "excluded_reason": f"only {n_kept} non-degenerate pairs",
            }
        else:
            bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
            entry = {
                "real_mean": float(real_scores.mean()), "random_mean": float(random_scores.mean()),
                "n_degenerate_real": int(real_degenerate.sum()), "n_degenerate_random": int(random_degenerate.sum()),
                **bootstrap,
            }

        results["by_layer"][layer] = entry
        sig_marker = " <-- SIGNIFICANT" if entry["significant_at_95pct"] else ""
        print(f"layer {layer:2d}: real={entry['real_mean']:.4f} random={entry['random_mean']:.4f} "
              f"diff={entry.get('point_estimate')} n={entry['n']}{sig_marker}", flush=True)

    significant_layers = [l for l, e in results["by_layer"].items() if e["significant_at_95pct"]]
    results["significant_layers"] = significant_layers
    print(f"\n=== LAYERS WHERE SINGLE-LAYER STEERING SIGNIFICANTLY BEATS RANDOM CONTROL (alpha={SWEEP_ALPHA}) ===", flush=True)
    print(significant_layers, flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
