"""L57: activation-steering run for soluble expression yield (eSol), reusing
L42's exact difference-of-means construction, generation setup, degeneracy
filter, and paired-bootstrap significance test. Only the target dataset
(eSol, src/l38/data_cache/expression/esol_clean.csv) and the scoring proxy
(absolute charge average, src/l38/l57_expression_yield_steering.py) are new.

Judged against docs/L50_CAPABILITY_GAIN_PROTOCOL.md's 6 criteria:
  1 -- real direction beats a matched-norm random direction, direct paired
       bootstrap, plus an unsteered baseline arm for reference.
  2 -- dose-response across SAFE_ALPHAS only (see SAFE_ALPHAS below).
  3 -- residue-exclusion robustness check at best_alpha.
  4 -- proxy pre-validated against eSol's real labels BEFORE this run:
       r=+0.305 full set / +0.337 held-out test split. See the proxy module's
       docstring for the full validation, the GRAVY orthogonality check, and
       the length-confound analysis.
  5 -- N/A: there is no prior technique for THIS property (L42's
       thermostability baseline steers a different property with a different
       proxy, so a head-to-head number would be meaningless). Criteria
       1-4 + 6 are the operative PASS bar, same reasoning as L51.
  6 -- n>=150 eval sequences per L50's rule for a NEW target property
       (L43's null only resolved cleanly at n~288); 233 available here.

Run this on a machine with ESM2-650M; ~12 arms x 150+ sequences of forward
passes. Not run as part of any test suite.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    dose_response_is_monotonic_then_collapsing,
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
)
from src.l38.l57_expression_yield_steering import (
    expression_yield_proxy,
    expression_yield_proxy_excluding,
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "expression" / "esol_clean.csv"
OUT_DIR = Path(__file__).resolve().parent / "l57_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42/L43/L51/L52
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 150  # L50 criterion 6's n>=150 floor for a new-property claim
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # L42's empirically-derived range
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30

# L42's established constraint (docs/L42_STEERING_REPRO.md "Honest verdict";
# re-affirmed in docs/L52_LAYER_SUBSET_STEERING.md's "Critical correction"):
# alpha >= 1.0 collapses this eval methodology (single-shot argmax mask-fill)
# into degenerate output regardless of whether the steering vector is doing
# anything real, so alpha >= 1.0 is excluded from best_alpha selection and
# from the dose-response check. L52's first draft picked alpha=2.0 and got a
# spurious PASS purely from two arms collapsing at different rates.
SAFE_ALPHAS = (0.1, 0.25, 0.5)


class MultiLayerSteeringHook:
    """Identical to L42/L43/L51/L52's hook -- adds alpha*direction to a
    layer's output, renormalized to preserve the original per-token
    activation norm (so no effect can come from simply inflating magnitude).
    """

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


def score_expression_yield(sequences):
    return np.array([expression_yield_proxy(seq) for seq in sequences])


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    # Use eSol's OWN split assignment: vectors from train, eval from the
    # disjoint valid+test held-out sequences -- a real held-out set, not a
    # resample of the vector-building pool.
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    held_df = df[df["split"].isin(["valid", "test"])].reset_index(drop=True)
    print(f"train: {len(train_df)}, held-out: {len(held_df)}", flush=True)

    train_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # High/low groups by the REAL experimental soluble-fraction label
    # (percentile split on the label, never on the proxy -- otherwise the
    # steering vector and the scorer would be circular).
    labels = train_shuffled["label"].astype(float).values
    low_threshold = np.percentile(labels, 20.0)
    high_threshold = np.percentile(labels, 80.0)
    low_group = train_shuffled[train_shuffled["label"].astype(float) <= low_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    high_group = train_shuffled[train_shuffled["label"].astype(float) >= high_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} low-yield, {len(high_group)} high-yield", flush=True)
    assert len(low_group) == N_VECTOR_SEQS_PER_GROUP and len(high_group) == N_VECTOR_SEQS_PER_GROUP, (
        f"need {N_VECTOR_SEQS_PER_GROUP} per group, got {len(low_group)}/{len(high_group)}"
    )

    # Eval sequences: LOW-yield held-out sequences -- steering should push
    # them UP, mirroring L42's low-Tm and L51's low-resistance eval choice.
    held_shuffled = held_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    held_median = np.percentile(held_shuffled["label"].astype(float).values, 50.0)
    eval_sequences = held_shuffled[held_shuffled["label"].astype(float) <= held_median]["sequence"].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences (low-yield, held out): {len(eval_sequences)}", flush=True)
    assert len(eval_sequences) >= N_EVAL_SEQS, (
        f"L50 criterion 6 requires >={N_EVAL_SEQS} eval sequences for a new property, got {len(eval_sequences)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-yield group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-yield group (all layers)...", flush=True)
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
            handles.append(model.esm.encoder.layer[layer].register_forward_hook(MultiLayerSteeringHook(vec, alpha)))
        return handles

    def generate_then_score(vectors, alpha):
        generated = []
        handles = apply_hooks(vectors, alpha) if alpha != 0.0 else []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            for h in handles:
                h.remove()
        return generated, score_expression_yield(generated)

    def score_arm(vectors, alpha):
        generated, scores = generate_then_score(vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, "
          f"degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)

    real_by_alpha = {}
    random_by_alpha = {}
    for alpha in ALPHAS:
        print(f"\n=== real_direction, alpha={alpha} ===", flush=True)
        real_by_alpha[alpha] = score_arm(steering_vectors, alpha)
        _, s, d = real_by_alpha[alpha]
        print(f"mean score: {s.mean():.4f}, degenerate: {d.sum()}/{len(d)}", flush=True)

        print(f"=== random_control, alpha={alpha} ===", flush=True)
        random_by_alpha[alpha] = score_arm(random_vectors, alpha)
        _, s, d = random_by_alpha[alpha]
        print(f"mean score: {s.mean():.4f}, degenerate: {d.sum()}/{len(d)}", flush=True)

    results = {
        "dataset": str(DATA_PATH),
        "proxy": "absolute_charge_average (pre-validated r=+0.305 full / +0.337 held-out test)",
        "baseline": {
            "mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()),
            "n": len(baseline_scores), "n_degenerate": int(baseline_degenerate.sum()),
        },
        "real_direction": {}, "random_control": {},
    }

    # Criterion 1: direct paired bootstrap, real vs random (NOT two separate
    # vs.-baseline tests), on pairs where neither arm nor baseline degenerated.
    real_vs_random_by_alpha = {}
    for alpha in ALPHAS:
        _, real_scores, real_degenerate = real_by_alpha[alpha]
        _, random_scores, random_degenerate = random_by_alpha[alpha]
        results["real_direction"][alpha] = {"mean": float(real_scores.mean()), "n_degenerate": int(real_degenerate.sum())}
        results["random_control"][alpha] = {"mean": float(random_scores.mean()), "n_degenerate": int(random_degenerate.sum())}

        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())
        if n_kept < MIN_NONDEGENERATE_PAIRS:
            real_vs_random_by_alpha[alpha] = {
                "point_estimate": None, "ci_lower": None, "ci_upper": None,
                "significant_at_95pct": False, "n": n_kept,
                "excluded_reason": f"only {n_kept} non-degenerate pairs, below MIN_NONDEGENERATE_PAIRS={MIN_NONDEGENERATE_PAIRS}",
            }
            print(f"\nalpha={alpha}: SKIPPED -- only {n_kept} non-degenerate pairs", flush=True)
            continue
        bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
        bootstrap["pct_sequences_real_beats_random"] = float((real_scores[keep] > random_scores[keep]).mean())
        real_vs_random_by_alpha[alpha] = bootstrap
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] "
              f"sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={bootstrap['pct_sequences_real_beats_random']:.3f}", flush=True)

    # Criterion 2: dose-response over SAFE_ALPHAS only.
    valid_alphas = [a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a]["point_estimate"] is not None]
    dose_response_ok = (
        dose_response_is_monotonic_then_collapsing(
            valid_alphas, [real_vs_random_by_alpha[a]["point_estimate"] for a in valid_alphas]
        )
        if len(valid_alphas) >= 3  # L50 criterion 2 requires >=3 sweep points
        else False
    )

    # Criterion 3: residue-exclusion robustness at best_alpha, restricted to
    # SAFE_ALPHAS (an alpha>=1.0 winner would be a collapse artifact).
    best_alpha = max(
        (a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a].get("significant_at_95pct")),
        key=lambda a: real_vs_random_by_alpha[a]["point_estimate"],
        default=None,
    )
    robustness_check = None
    if best_alpha is not None:
        real_generated, _, real_degenerate = real_by_alpha[best_alpha]
        random_generated, _, random_degenerate = random_by_alpha[best_alpha]
        counts = Counter()
        for seq, base in zip(real_generated, baseline_generated):
            for a, b in zip(seq, base):
                if a != b:
                    counts[a] += 1
        top_residues = frozenset(r for r, _ in counts.most_common(2))
        print(f"\ndominant substituted residues at alpha={best_alpha}: {counts.most_common(5)}", flush=True)

        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        real_excl = np.array([expression_yield_proxy_excluding(s, top_residues) for s in np.array(real_generated)[keep]])
        random_excl = np.array([expression_yield_proxy_excluding(s, top_residues) for s in np.array(random_generated)[keep]])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl, real_excl, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha,
            "excluded_residues": sorted(top_residues),
            "substitution_counts_top5": counts.most_common(5),
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): diff={excl_bootstrap['point_estimate']:.4f} "
              f"[{excl_bootstrap['ci_lower']:.4f}, {excl_bootstrap['ci_upper']:.4f}] "
              f"sig={excl_bootstrap['significant_at_95pct']}", flush=True)

    crit1 = best_alpha is not None
    crit2 = dose_response_ok
    crit3 = robustness_check is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
    crit4 = True  # proxy pre-validated against eSol labels before this run (see proxy module docstring)
    crit6 = len(eval_sequences) >= N_EVAL_SEQS
    criteria = {
        "1_beats_random_control": crit1,
        "2_dose_response": crit2,
        "3_residue_robust": crit3,
        "4_proxy_pre_validated": crit4,
        "5_beats_prior_technique": "N/A -- no prior technique for this property",
        "6_adequately_powered": crit6,
    }
    if not crit1:
        decision = "KILL"
    elif not crit3:
        decision = "AMBIGUOUS"  # significant but artifact-driven -- L43's exact failure mode
    elif crit2 and crit4 and crit6:
        decision = "PASS"
    else:
        decision = "AMBIGUOUS"

    verdict = {
        "criteria": criteria,
        "decision": decision,
        "best_alpha": best_alpha,
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "robustness_check": robustness_check,
    }
    print("\n=== L57 VERDICT (expression yield / eSol, degenerate-filtered, "
          "paired-bootstrapped, residue-exclusion-checked) ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results["verdict"] = verdict
    results["raw_scores"] = {
        "baseline": baseline_scores.tolist(),
        **{f"real__{a}": real_by_alpha[a][1].tolist() for a in ALPHAS},
        **{f"random__{a}": random_by_alpha[a][1].tolist() for a in ALPHAS},
    }
    results["raw_sequences"] = {
        "baseline": baseline_generated,
        **{f"real__{a}": real_by_alpha[a][0] for a in ALPHAS},
        **{f"random__{a}": random_by_alpha[a][0] for a in ALPHAS},
    }
    results["eval_sequences"] = eval_sequences
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
