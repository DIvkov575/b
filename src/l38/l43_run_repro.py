"""L43: extend the L42 activation-steering harness (validated on
thermostability, docs/L42_STEERING_REPRO.md) to a second target property --
solubility -- to test whether the reproduction generalizes beyond
thermostability specifically. Same model, same difference-of-means vector
construction, same degeneracy filter, same paired-bootstrap significance
test as L42 -- only the target dataset (hazemessam/solubility, real
soluble/insoluble labels) and the scoring function (GRAVY hydropathy proxy,
src/l38/l43_solubility_steering.py) are new.

See docs/L43_SOLUBILITY_STEERING.md for the full protocol and PASS/KILL rule.
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
from src.l38.l43_solubility_steering import solubility_proxy

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
TRAIN_PATH = Path(__file__).resolve().parent / "data_cache" / "solubility" / "train.csv"
OUT_DIR = Path(__file__).resolve().parent / "l43_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42's convention
N_VECTOR_SEQS_PER_GROUP = 150  # sequences used to BUILD the steering vector
N_EVAL_SEQS = 60  # held-out sequences steered/scored, disjoint from vector-building set
# Reuse L42's alpha grid and mask fraction unchanged -- both were empirically
# derived against THIS model (ESM2-650M) and THIS generation setup
# (single-shot argmax mask-fill), not against thermostability specifically,
# so there's no a priori reason they'd need to differ for solubility. If the
# collapse point turns out to differ, that itself is a finding worth noting.
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30  # see docs/L42_STEERING_REPRO.md RESULTS v1 for
# why this guard exists: a handful of sub-threshold-collapsed survivors can
# fake a significant result if trusted without a minimum sample size.


class MultiLayerSteeringHook:
    """Adds alpha*direction[layer] to a specific transformer layer's output,
    renormalized to preserve the original per-token activation norm -- per
    Huang et al.'s method. Identical to L42's hook; copied rather than
    imported only because l42_run_repro.py isn't meant to be a shared
    library module (it's a standalone reproduction script)."""

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
    """Returns dict layer_idx -> [n_seqs, d_model] mean-pooled activations,
    for every transformer layer (excluding the embedding layer)."""
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
    """Mask most of the sequence, single-shot-predict the masked positions
    (argmax per position), return the generated sequence. Identical to
    L42's generate-then-score pattern (docs/L42_STEERING_REPRO.md) -- the
    hook is active during THIS forward pass; scoring always runs afterward
    on the unperturbed model."""
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


def score_solubility_proxy(sequences):
    """Independent solubility PROXY: GRAVY hydropathy (Kyte & Doolittle 1982),
    negated so higher = more soluble-like, consistent with L42's "higher
    score = better" sign convention. See src/l38/l43_solubility_steering.py.
    """
    return np.array([solubility_proxy(seq) for seq in sequences])


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(TRAIN_PATH)
    df = df[df["sequences"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]

    # Real experimental labels (1=soluble, 0=insoluble) -- use directly as
    # the high/low split rather than a percentile-of-a-continuous-score split
    # (L42's approach), since this dataset's ground truth is already binary.
    insoluble_pool = vector_pool[vector_pool["labels"] == 0]["sequences"].tolist()
    soluble_pool = vector_pool[vector_pool["labels"] == 1]["sequences"].tolist()
    low_group = insoluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    high_group = soluble_pool[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} insoluble, {len(high_group)} soluble", flush=True)

    # Eval sequences: insoluble-labeled, held out from vector construction --
    # steering should push a low-solubility sequence toward higher-
    # solubility-like activations, mirroring L42's low-Tm eval-sequence choice.
    eval_insoluble = eval_pool[eval_pool["labels"] == 0]["sequences"].tolist()
    eval_sequences = eval_insoluble[:N_EVAL_SEQS]
    print(f"eval sequences (insoluble, held out from vector construction): {len(eval_sequences)}", flush=True)

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
    print(f"built {len(steering_vectors)} per-layer difference-of-means steering vectors", flush=True)

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
        if alpha == 0.0:
            continue
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

    nonzero_alphas = [a for a in ALPHAS if a != 0.0]
    real_vs_random_by_alpha = {}
    for alpha in nonzero_alphas:
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
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] sig={bootstrap['significant_at_95pct']}", flush=True)

    verdict = {
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "decision": "PASS" if any(v["significant_at_95pct"] for v in real_vs_random_by_alpha.values()) else "INCONCLUSIVE",
    }

    print("\n=== L43 VERDICT (degenerate-filtered, paired-bootstrapped) ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)

    results["verdict"] = verdict
    results["all_sequences"] = all_sequences
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
