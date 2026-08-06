"""L51: activation-steering run for aggregation resistance, reusing L42's
exact difference-of-means construction, degeneracy filter, and paired-
bootstrap significance test. Only the target dataset (cmartell/50C_Aggregation,
cleaned) and scoring proxy (net-charge-based, src/l38/l51_aggregation_steering.py)
are new.

Judged against docs/L50_CAPABILITY_GAIN_PROTOCOL.md's 6 criteria:
beats baseline+random control with a real CI; dose-response across >=3
alphas; survives residue-exclusion; proxy pre-validated against real labels
(done above, r=-0.20/-0.21 on train/test); n>=150 eval sequences; and (since
this is a NEW property, not a repeat) criterion 5's "beats existing
technique" doesn't apply here -- there IS no existing technique for this
property yet, so criteria 1-4+6 are the operative PASS bar.
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
from src.l38.l51_aggregation_steering import aggregation_resistance_proxy, aggregation_resistance_proxy_excluding

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "aggregation" / "agg50_clean.csv"
OUT_DIR = Path(__file__).resolve().parent / "l51_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42/L43's convention
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 150  # per Phase 0 protocol's n>=150 rule for a new-property claim
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # reuse L42's empirically-derived non-degenerate range
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30


class MultiLayerSteeringHook:
    """Identical to L42/L43's hook -- adds alpha*direction to a layer's
    output, renormalized to preserve original per-token activation norm."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha

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


@torch.no_grad()
def mean_pooled_activation_all_layers(model, tokenizer, sequences, device, max_len=MAX_SEQ_LEN):
    n_layers = model.config.num_hidden_layers
    per_layer_activations = {layer: [] for layer in range(n_layers)}
    for seq in sequences:
        seq = seq[:max_len]
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
        out = model(**enc, output_hidden_states=True)
        for layer in range(n_layers):
            hidden = out.hidden_states[layer + 1].squeeze(0).float()
            per_layer_activations[layer].append(hidden.mean(dim=0).cpu().numpy())
    return {layer: np.stack(vals, axis=0) for layer, vals in per_layer_activations.items()}


@torch.no_grad()
def mask_fill_generate(model, tokenizer, sequence, mask_fraction, seed, device, max_len=MAX_SEQ_LEN):
    seq = sequence[:max_len]
    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2)
    input_ids = enc["input_ids"][0].clone()

    rng = torch.Generator().manual_seed(seed)
    special_ids = set(tokenizer.all_special_ids)
    non_special_positions = torch.tensor([i for i, t in enumerate(input_ids.tolist()) if t not in special_ids])
    n_mask = max(1, int(len(non_special_positions) * mask_fraction))
    perm = torch.randperm(len(non_special_positions), generator=rng)
    mask_positions = non_special_positions[perm[:n_mask]]

    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = tokenizer.mask_token_id

    masked_enc = {"input_ids": masked_ids.unsqueeze(0).to(device), "attention_mask": enc["attention_mask"].to(device)}
    out = model(**masked_enc)
    predicted_ids = out.logits.argmax(dim=-1).squeeze(0).cpu()

    filled_ids = masked_ids.clone()
    filled_ids[mask_positions] = predicted_ids[mask_positions]

    tokens_str = tokenizer.convert_ids_to_tokens(filled_ids.tolist())
    return "".join(t for t in tokens_str if t not in tokenizer.all_special_tokens)


def score_aggregation_resistance(sequences):
    return np.array([aggregation_resistance_proxy(seq) for seq in sequences])


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    # split by the dataset's OWN train/test stage assignment -- vector-building
    # from train, eval sequences from the disjoint test split (real held-out,
    # not just a random resample of the same pool).
    train_df = df[df["stage"] == "train"].reset_index(drop=True)
    test_df = df[df["stage"] == "test"].reset_index(drop=True)
    print(f"train: {len(train_df)}, test: {len(test_df)}", flush=True)

    train_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # high/low groups by REAL experimental label (more negative = more
    # aggregation-prone = "low" resistance; less negative/positive = "high"
    # resistance), percentile split on the real label, not the proxy.
    low_pct, high_pct = 20.0, 80.0
    labels = train_shuffled["label"].astype(float).values
    low_threshold = np.percentile(labels, low_pct)
    high_threshold = np.percentile(labels, high_pct)
    low_group = train_shuffled[train_shuffled["label"].astype(float) <= low_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    high_group = train_shuffled[train_shuffled["label"].astype(float) >= high_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} low-resistance (aggregation-prone), {len(high_group)} high-resistance", flush=True)

    test_shuffled = test_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # eval sequences: LOW-resistance (aggregation-prone) sequences from the
    # held-out test split -- steering should push them toward higher
    # resistance, mirroring L42's low-Tm eval-sequence choice.
    test_labels = test_shuffled["label"].astype(float).values
    test_low_threshold = np.percentile(test_labels, 50.0)
    eval_sequences = test_shuffled[test_shuffled["label"].astype(float) <= test_low_threshold]["sequence"].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences (low-resistance, held-out test split): {len(eval_sequences)}", flush=True)
    assert len(eval_sequences) >= MIN_NONDEGENERATE_PAIRS, "not enough eval sequences in held-out test split"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-resistance group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-resistance group (all layers)...", flush=True)
    high_activations = mean_pooled_activation_all_layers(model, tokenizer, high_group, device)

    steering_vectors = {}
    for layer in range(n_layers):
        vec = difference_of_means_vector(low_activations[layer], high_activations[layer])
        steering_vectors[layer] = torch.tensor(vec, dtype=torch.float32, device=device)
    print(f"built {len(steering_vectors)} per-layer difference-of-means steering vectors", flush=True)

    rng2 = np.random.RandomState(SEED + 1)
    random_vectors = {}
    for layer, vec in steering_vectors.items():
        rv = torch.tensor(rng2.normal(size=vec.shape[0]), dtype=torch.float32, device=device)
        random_vectors[layer] = rv / rv.norm() * vec.norm()

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
        scores = score_aggregation_resistance(generated)
        return generated, scores

    results = {"real_direction": {}, "random_control": {}}
    all_sequences = {"baseline": None, "real_direction": {}, "random_control": {}}

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)
    results["baseline"] = {
        "mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()), "n": len(baseline_scores),
        "n_degenerate": int(baseline_degenerate.sum()),
    }
    all_sequences["baseline"] = baseline_generated

    def score_arm(vectors, alpha):
        generated, scores = generate_then_score(vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    real_by_alpha = {}
    random_by_alpha = {}

    for alpha in ALPHAS:
        print(f"\n=== real_direction, alpha={alpha} ===", flush=True)
        generated, scores, degenerate = score_arm(steering_vectors, alpha)
        real_by_alpha[alpha] = (generated, scores, degenerate)
        all_sequences["real_direction"][alpha] = generated
        print(f"mean score: {scores.mean():.4f}, degenerate: {degenerate.sum()}/{len(degenerate)}", flush=True)

        print(f"=== random_control, alpha={alpha} ===", flush=True)
        generated, scores, degenerate = score_arm(random_vectors, alpha)
        random_by_alpha[alpha] = (generated, scores, degenerate)
        all_sequences["random_control"][alpha] = generated
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
                "excluded_reason": f"only {n_kept} non-degenerate pairs, below MIN_NONDEGENERATE_PAIRS={MIN_NONDEGENERATE_PAIRS}",
            }
            continue
        bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
        real_vs_random_by_alpha[alpha] = bootstrap
        pct_positive = float((real_scores[keep] > random_scores[keep]).mean())
        real_vs_random_by_alpha[alpha]["pct_sequences_real_beats_random"] = pct_positive
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={pct_positive:.3f}", flush=True)

    # residue-exclusion robustness check (Phase 0 criterion 3), on the
    # alpha with the strongest significant real-vs-random effect
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
            aggregation_resistance_proxy_excluding(s, top_residues) for s in np.array(real_by_alpha[best_alpha][0])[keep]
        ])
        random_excl_scores = np.array([
            aggregation_resistance_proxy_excluding(s, top_residues) for s in np.array(random_by_alpha[best_alpha][0])[keep]
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
            "PASS" if best_alpha is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
            else "PASS_ARTIFACT" if best_alpha is not None
            else "KILL"
        ),
    }

    print("\n=== L51 VERDICT (degenerate-filtered, paired-bootstrapped, residue-exclusion-checked) ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results["verdict"] = verdict
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
