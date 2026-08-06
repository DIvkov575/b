"""L45 follow-up: causal NECESSITY sweep, complementing L45's causal
SUFFICIENCY sweep (docs/L45_LAYER_SWEEP.md).

L45 (original) asked: does steering ONE layer alone reproduce part of the
all-layers effect? (sufficiency). This asks the complementary question: if
you steer ALL layers EXCEPT one (leave-one-out), does removing that one
layer's contribution measurably shrink the full 33-layer effect relative to
steering all 33? A layer that's causally NECESSARY for the full effect
should show a bigger drop when left out than a layer that's just "along for
the ride."

Design: L42's full-33-layer steering (alpha=0.25, its cleanest clean signal)
is the reference effect. For each layer L, apply the SAME steering vectors
to all 32 OTHER layers (L itself gets alpha=0, i.e. untouched) and measure
the real-vs-random effect exactly as L42/L45 do. A layer whose omission
causes the biggest drop in effect size (relative to the full-33 baseline)
is the most causally necessary.

Second question (per user request): does the same depth-weighting pattern
L45 found for thermostability also show up if this necessity-sweep method
is applied to L43's solubility vectors -- even though L43's all-layers
solubility effect was itself an artifact (docs/L43_SOLUBILITY_STEERING.md)?
If leave-one-out on the solubility vectors shows no coherent depth pattern
at all, that's consistent with "there was no real effect to localize in the
first place." Included as a built-in negative control on the METHOD, not
just a repeat of L43.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_run_repro import (
    MODEL_NAME as THERMO_MODEL_NAME,
    DATA_PATH as THERMO_DATA_PATH,
    MAX_SEQ_LEN,
    N_VECTOR_SEQS_PER_GROUP,
    N_EVAL_SEQS,
    MASK_FRACTION,
    SEED,
    mean_pooled_activation_all_layers,
    mask_fill_generate,
    score_thermostability_proxy,
)
from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
    split_by_percentile,
    layer_effects_sign_test,
)
from src.l38.l43_run_repro import TRAIN_PATH as SOLUBILITY_TRAIN_PATH, score_solubility_proxy

SWEEP_ALPHA = 0.25  # L42's validated clean signal
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30


class LeaveOneOutSteeringHook:
    """Adds alpha*direction to a layer's output, EXCEPT for `excluded_layer`,
    which is left untouched (alpha implicitly 0 there) -- registered per
    layer, one hook instance carries its own layer index and skips
    application when that index matches the layer being left out."""

    def __init__(self, direction: torch.Tensor, alpha: float, this_layer: int, excluded_layer: int):
        self.direction = direction
        self.alpha = alpha if this_layer != excluded_layer else 0.0

    def __call__(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        original_norm = hidden.norm(dim=-1, keepdim=True)
        perturbed = hidden + self.alpha * self.direction
        perturbed_norm = perturbed.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        renormalized = perturbed * (original_norm / perturbed_norm)
        if isinstance(output, tuple):
            return (renormalized,) + output[1:]
        return renormalized


def run_necessity_sweep(
    property_name, model_name, data_path, load_data_fn, score_fn, out_path, encoder_attr="esm.encoder.layer",
):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[{property_name}] device: {device}", flush=True)

    low_group, high_group, eval_sequences = load_data_fn(data_path)
    print(f"[{property_name}] vector groups: {len(low_group)} low, {len(high_group)} high; eval: {len(eval_sequences)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"[{property_name}] model loaded: {model_name}, {n_layers} layers", flush=True)

    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
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

    layer_modules = model
    for attr in encoder_attr.split("."):
        layer_modules = getattr(layer_modules, attr)

    def generate_with_hooks(vectors, excluded_layer, alpha):
        handles = []
        for layer, vec in vectors.items():
            hook = LeaveOneOutSteeringHook(vec, alpha, layer, excluded_layer)
            handles.append(layer_modules[layer].register_forward_hook(hook))
        generated = []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            for h in handles:
                h.remove()
        scores = score_fn(generated)
        return generated, scores

    print(f"\n[{property_name}] === baseline (no hooks) ===", flush=True)
    baseline_generated = [
        mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device)
        for i, seq in enumerate(eval_sequences)
    ]
    baseline_scores = score_fn(baseline_generated)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)

    # reference: ALL 33 layers steered (excluded_layer=-1, matches nothing)
    print(f"\n[{property_name}] === reference: all {n_layers} layers steered (alpha={SWEEP_ALPHA}) ===", flush=True)
    full_real_generated, full_real_scores = generate_with_hooks(steering_vectors, excluded_layer=-1, alpha=SWEEP_ALPHA)
    full_random_generated, full_random_scores = generate_with_hooks(random_vectors, excluded_layer=-1, alpha=SWEEP_ALPHA)
    full_real_degenerate = np.array([is_degenerate_sequence(s) for s in full_real_generated])
    full_random_degenerate = np.array([is_degenerate_sequence(s) for s in full_random_generated])
    full_keep = ~full_real_degenerate & ~full_random_degenerate & ~baseline_degenerate
    full_n_kept = int(full_keep.sum())
    if full_n_kept >= MIN_NONDEGENERATE_PAIRS:
        full_bootstrap = paired_bootstrap_mean_diff(full_random_scores[full_keep], full_real_scores[full_keep], n_boot=N_BOOT, seed=SEED)
    else:
        full_bootstrap = {"point_estimate": None, "n": full_n_kept, "excluded_reason": "too few non-degenerate pairs"}
    full_effect = full_bootstrap["point_estimate"]
    print(f"full-{n_layers}-layer effect: {full_effect} (n={full_n_kept})", flush=True)

    results = {
        "property": property_name, "sweep_alpha": SWEEP_ALPHA,
        "full_effect_all_layers": full_bootstrap,
        "by_excluded_layer": {},
    }

    for excluded_layer in range(n_layers):
        real_generated, real_scores = generate_with_hooks(steering_vectors, excluded_layer, SWEEP_ALPHA)
        random_generated, random_scores = generate_with_hooks(random_vectors, excluded_layer, SWEEP_ALPHA)
        real_degenerate = np.array([is_degenerate_sequence(s) for s in real_generated])
        random_degenerate = np.array([is_degenerate_sequence(s) for s in random_generated])
        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())

        if n_kept < MIN_NONDEGENERATE_PAIRS:
            entry = {
                "point_estimate": None, "n": n_kept,
                "excluded_reason": f"only {n_kept} non-degenerate pairs", "drop_from_full": None,
            }
        else:
            bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
            drop = (full_effect - bootstrap["point_estimate"]) if full_effect is not None else None
            entry = {**bootstrap, "drop_from_full": drop}

        results["by_excluded_layer"][excluded_layer] = entry
        print(f"exclude layer {excluded_layer:2d}: effect={entry.get('point_estimate')} "
              f"drop_from_full={entry.get('drop_from_full')} n={entry['n']}", flush=True)

    drops = [
        e["drop_from_full"] for e in results["by_excluded_layer"].values() if e.get("drop_from_full") is not None
    ]
    if drops:
        sign_test = layer_effects_sign_test(drops)
        results["drop_sign_test"] = sign_test
        print(f"\n[{property_name}] sign test on drops (positive drop = layer is necessary): {sign_test}", flush=True)

        sorted_layers = sorted(
            ((l, e["drop_from_full"]) for l, e in results["by_excluded_layer"].items() if e.get("drop_from_full") is not None),
            key=lambda x: -x[1],
        )
        results["most_necessary_layers"] = sorted_layers[:10]
        print(f"[{property_name}] top-10 most necessary layers (biggest drop when excluded): {sorted_layers[:10]}", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[{property_name}] saved to {out_path}", flush=True)
    return results


def load_thermostability_data(data_path):
    df = pd.read_csv(data_path)
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
    return low_group, high_group, eval_sequences


def load_solubility_data(data_path):
    df = pd.read_csv(data_path)
    df = df[df["sequences"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]
    insoluble_pool = vector_pool[vector_pool["labels"] == 0]["sequences"].tolist()
    soluble_pool = vector_pool[vector_pool["labels"] == 1]["sequences"].tolist()
    low_group = insoluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    high_group = soluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    eval_insoluble = eval_pool[eval_pool["labels"] == 0]["sequences"].tolist()
    eval_sequences = eval_insoluble[:N_EVAL_SEQS]
    return low_group, high_group, eval_sequences


def main():
    out_dir = Path(__file__).resolve().parent
    run_necessity_sweep(
        "thermostability", THERMO_MODEL_NAME, THERMO_DATA_PATH, load_thermostability_data,
        score_thermostability_proxy, out_dir / "l45_necessity_sweep_thermostability_out.json",
    )
    run_necessity_sweep(
        "solubility", THERMO_MODEL_NAME, SOLUBILITY_TRAIN_PATH, load_solubility_data,
        score_solubility_proxy, out_dir / "l45_necessity_sweep_solubility_out.json",
    )


if __name__ == "__main__":
    main()
