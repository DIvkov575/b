"""L52: Phase 1 of docs/L50_CAPABILITY_GAIN_PROTOCOL.md -- does restricting
activation steering to ONLY the 5 layers L45's leave-one-out sweep found
causally necessary for thermostability (18, 23, 25, 30, 31, out of all 33)
preserve the steering effect, instead of needing all 33 layers touched
(L42's original config)?

Reuses L42's exact data loading/splitting/eval-sequence selection (same
SEED, same percentile split, same N_VECTOR_SEQS_PER_GROUP/N_EVAL_SEQS, same
IVYWREL proxy -- already validated, not a new-property claim) so the
all-33-layer arm recomputed here is a same-run, same-eval-set replication of
L42's PASS, not a reused cached number from a different eval draw.

Judged against all 6 L50 criteria:
  1/2/3/4 -- identical in kind to L42's own bar, run fresh for the 5-layer
    subset arm (beats baseline+random control with real CI; dose-response
    across the same 5 alphas; residue-exclusion robustness; IVYWREL proxy
    inherited pre-validated).
  5 -- NEW here: head-to-head, same alpha, same eval sequences, same seed,
    subset-5 vs all-33 (L42's existing best-known config for this property).
    Not-significantly-worse is the bar, since the scientific claim is
    "5 layers suffice," not "5 layers is a bigger effect than 33."
  6 -- n=60 eval sequences, matching L42's existing guard (>=30 pairs).
    NOT bumped to n>=150: that rule is scoped to claims about a NEW target
    property (per L50's own criterion-6 text); thermostability is L42's
    already-established property, only the layer-scope of the mechanism is
    new here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    dose_response_is_monotonic_then_collapsing,
    is_degenerate_sequence,
    ivywrel_fraction,
    paired_bootstrap_mean_diff,
    split_by_percentile,
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "meltome" / "mixed_split.csv"
OUT_DIR = Path(__file__).resolve().parent / "l52_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 60  # matches L42 exactly -- see module docstring on criterion 6
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # identical range to L42
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30

# L45's leave-one-out causal-necessity sweep top-5 layers for thermostability
# (docs/L45_LAYER_SWEEP.md / l45_necessity_sweep_thermostability_out.json),
# ranked by drop_from_full when excluded: 31, 30, 25, 18, 23.
NECESSARY_LAYERS = frozenset({18, 23, 25, 30, 31})

# L42's own established constraint (docs/L42_STEERING_REPRO.md, "Honest
# verdict" section): alpha >= 1.0 degenerates this eval methodology
# (single-shot argmax mask-fill) into poly-leucine collapse independent of
# whether the steering vector/technique is doing anything real, so any
# alpha >= 1.0 comparison is untrustworthy regardless of what its bootstrap
# CI says. best_alpha selection below is restricted to this range -- picking
# outside it previously produced a spurious PASS (subset5 "beat" all33 at
# alpha=2.0 only because all33 had FULLY collapsed there, 0/60 non-degenerate,
# while subset5 hadn't yet -- an artifact of collapse-order, not a real
# advantage).
SAFE_ALPHAS = (0.1, 0.25, 0.5)


class MultiLayerSteeringHook:
    """Identical to L42/L43/L51's hook -- adds alpha*direction to a layer's
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


def score_thermostability_proxy(sequences):
    return np.array([ivywrel_fraction(seq) for seq in sequences])


def ivywrel_fraction_excluding(sequence: str, excluded: frozenset) -> float:
    kept = [c for c in sequence if c not in excluded]
    if len(kept) == 0:
        raise ValueError("ivywrel_fraction_excluding: sequence is empty after exclusion")
    from src.l38.l42_steering_repro import IVYWREL_RESIDUES
    residues = IVYWREL_RESIDUES - excluded
    return sum(1 for c in kept if c in residues) / len(kept)


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]

    low_group, high_group = split_by_percentile(
        vector_pool["sequence"].tolist(), vector_pool["label"].values, low_pct=20.0, high_pct=80.0
    )
    low_group = low_group[:N_VECTOR_SEQS_PER_GROUP]
    high_group = high_group[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} low-Tm, {len(high_group)} high-Tm", flush=True)

    eval_pool_sorted = eval_pool.sort_values("label")
    eval_sequences = eval_pool_sorted["sequence"].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences (low-Tm, held out from vector construction): {len(eval_sequences)}", flush=True)

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
    print(f"built {len(steering_vectors)} per-layer difference-of-means steering vectors", flush=True)

    rng2 = np.random.RandomState(SEED + 1)
    random_vectors = {
        layer: torch.tensor(rng2.normal(size=vec.shape[0]), dtype=torch.float32, device=device)
        for layer, vec in steering_vectors.items()
    }
    for layer in random_vectors:
        real_norm = steering_vectors[layer].norm()
        random_vectors[layer] = random_vectors[layer] / random_vectors[layer].norm() * real_norm

    def apply_hooks(vectors, alpha, layer_scope):
        handles = []
        for layer in layer_scope:
            hook = MultiLayerSteeringHook(vectors[layer], alpha)
            handles.append(model.esm.encoder.layer[layer].register_forward_hook(hook))
        return handles

    def remove_hooks(handles):
        for h in handles:
            h.remove()

    def generate_then_score(vectors, alpha, layer_scope):
        generated = []
        handles = apply_hooks(vectors, alpha, layer_scope) if alpha != 0.0 else []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            remove_hooks(handles)
        scores = score_thermostability_proxy(generated)
        return generated, scores

    def score_arm(vectors, alpha, layer_scope):
        generated, scores = generate_then_score(vectors, alpha, layer_scope)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    all_layers = list(range(n_layers))
    subset_layers = sorted(NECESSARY_LAYERS)

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0, all_layers)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)

    configs = {"all33": all_layers, "subset5": subset_layers}
    arms = {}  # (config, direction, alpha) -> (generated, scores, degenerate)
    for config_name, layer_scope in configs.items():
        for alpha in ALPHAS:
            print(f"\n=== {config_name}, real_direction, alpha={alpha} ===", flush=True)
            arms[(config_name, "real", alpha)] = score_arm(steering_vectors, alpha, layer_scope)
            g, s, d = arms[(config_name, "real", alpha)]
            print(f"mean score: {s.mean():.4f}, degenerate: {d.sum()}/{len(d)}", flush=True)

            print(f"=== {config_name}, random_control, alpha={alpha} ===", flush=True)
            arms[(config_name, "random", alpha)] = score_arm(random_vectors, alpha, layer_scope)
            g, s, d = arms[(config_name, "random", alpha)]
            print(f"mean score: {s.mean():.4f}, degenerate: {d.sum()}/{len(d)}", flush=True)

    def bootstrap_pair(gen_a, scores_a, deg_a, gen_b, scores_b, deg_b):
        keep = ~deg_a & ~deg_b & ~baseline_degenerate
        n_kept = int(keep.sum())
        if n_kept < MIN_NONDEGENERATE_PAIRS:
            return {
                "point_estimate": None, "ci_lower": None, "ci_upper": None,
                "significant_at_95pct": False, "n": n_kept,
                "excluded_reason": f"only {n_kept} non-degenerate pairs, below MIN_NONDEGENERATE_PAIRS={MIN_NONDEGENERATE_PAIRS}",
            }
        result = paired_bootstrap_mean_diff(scores_a[keep], scores_b[keep], n_boot=N_BOOT, seed=SEED)
        result["pct_b_beats_a"] = float((scores_b[keep] > scores_a[keep]).mean())
        return result

    # Criteria 1+2: subset5 real vs subset5 random, per alpha
    subset5_vs_random = {}
    for alpha in ALPHAS:
        _, real_s, real_d = arms[("subset5", "real", alpha)]
        _, rand_s, rand_d = arms[("subset5", "random", alpha)]
        subset5_vs_random[alpha] = bootstrap_pair(
            arms[("subset5", "random", alpha)][0], rand_s, rand_d,
            arms[("subset5", "real", alpha)][0], real_s, real_d,
        )
        print(f"\nsubset5 real-vs-random alpha={alpha}: {subset5_vs_random[alpha]}", flush=True)

    # sanity replication: all33 real vs all33 random, per alpha (should mirror L42)
    all33_vs_random = {}
    for alpha in ALPHAS:
        _, real_s, real_d = arms[("all33", "real", alpha)]
        _, rand_s, rand_d = arms[("all33", "random", alpha)]
        all33_vs_random[alpha] = bootstrap_pair(
            arms[("all33", "random", alpha)][0], rand_s, rand_d,
            arms[("all33", "real", alpha)][0], real_s, real_d,
        )
        print(f"all33 real-vs-random alpha={alpha}: {all33_vs_random[alpha]}", flush=True)

    # Criterion 5: head-to-head, subset5-real vs all33-real, per alpha
    subset5_vs_all33 = {}
    for alpha in ALPHAS:
        _, s5_s, s5_d = arms[("subset5", "real", alpha)]
        _, a33_s, a33_d = arms[("all33", "real", alpha)]
        subset5_vs_all33[alpha] = bootstrap_pair(
            arms[("all33", "real", alpha)][0], a33_s, a33_d,
            arms[("subset5", "real", alpha)][0], s5_s, s5_d,
        )
        print(f"subset5-vs-all33 (head-to-head) alpha={alpha}: {subset5_vs_all33[alpha]}", flush=True)

    # Criterion 2: dose-response check on subset5's real-vs-random effect
    # curve, restricted to SAFE_ALPHAS -- alpha >= 1.0 is excluded by design
    # (see SAFE_ALPHAS docstring: that regime degenerates the eval
    # methodology itself, not a real dose-response point).
    valid_alphas = [a for a in SAFE_ALPHAS if subset5_vs_random[a]["point_estimate"] is not None]
    dose_response_ok = dose_response_is_monotonic_then_collapsing(
        valid_alphas, [subset5_vs_random[a]["point_estimate"] for a in valid_alphas]
    ) if len(valid_alphas) >= 2 else False

    # Criterion 3/5: best_alpha restricted to SAFE_ALPHAS for the same reason
    best_alpha = max(
        (a for a in SAFE_ALPHAS if subset5_vs_random[a].get("significant_at_95pct")),
        key=lambda a: subset5_vs_random[a]["point_estimate"],
        default=None,
    )
    robustness_check = None
    if best_alpha is not None:
        from collections import Counter
        real_generated, _, real_degenerate = arms[("subset5", "real", best_alpha)]
        random_generated, _, random_degenerate = arms[("subset5", "random", best_alpha)]
        counts = Counter()
        for seq, base in zip(real_generated, baseline_generated):
            for a, b in zip(seq, base):
                if a != b:
                    counts[a] += 1
        top_residues = frozenset(r for r, _ in counts.most_common(2))
        print(f"\ndominant substituted residues at alpha={best_alpha}: {counts.most_common(5)}", flush=True)

        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        real_excl_scores = np.array([
            ivywrel_fraction_excluding(s, top_residues) for s in np.array(real_generated)[keep]
        ])
        random_excl_scores = np.array([
            ivywrel_fraction_excluding(s, top_residues) for s in np.array(random_generated)[keep]
        ])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl_scores, real_excl_scores, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha, "excluded_residues": sorted(top_residues),
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): {excl_bootstrap}", flush=True)

    crit1 = best_alpha is not None
    crit2 = dose_response_ok
    crit3 = robustness_check is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
    crit4 = True  # IVYWREL inherited pre-validated from L42; not re-derived here
    # criterion 5: at the SAME alpha subset5 was judged on, subset5 must not be
    # significantly WORSE than all33 head-to-head (non-inferiority, not "beats")
    crit5 = (
        best_alpha is not None
        and not (
            subset5_vs_all33[best_alpha]["point_estimate"] is not None
            and subset5_vs_all33[best_alpha]["significant_at_95pct"]
            and subset5_vs_all33[best_alpha]["point_estimate"] < 0
        )
    )
    crit6 = True  # n=60 matches L42's existing guard; property not new, see docstring

    criteria = {"1_beats_controls": crit1, "2_dose_response": crit2, "3_residue_robust": crit3,
                "4_proxy_pre_validated": crit4, "5_non_inferior_to_all33": crit5, "6_adequately_powered": crit6}
    n_pass = sum(criteria.values())
    if n_pass == 6:
        decision = "PASS"
    elif not crit1 or not crit3:
        decision = "KILL"
    elif n_pass >= 4:
        decision = "AMBIGUOUS"
    else:
        decision = "KILL"

    verdict = {
        "criteria": criteria,
        "decision": decision,
        "best_alpha": best_alpha,
        "subset5_vs_random_by_alpha": subset5_vs_random,
        "all33_vs_random_by_alpha": all33_vs_random,
        "subset5_vs_all33_by_alpha": subset5_vs_all33,
        "robustness_check": robustness_check,
    }

    print("\n=== L52 VERDICT (Phase 1: layer-subset vs all-33-layer, thermostability) ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results = {
        "baseline": {"mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()),
                     "n": len(baseline_scores), "n_degenerate": int(baseline_degenerate.sum())},
        "verdict": verdict,
        "raw_scores": {
            f"{config}__{direction}__{alpha}": scores.tolist()
            for (config, direction, alpha), (gen, scores, deg) in arms.items()
        },
        "raw_sequences": {
            f"{config}__{direction}__{alpha}": gen
            for (config, direction, alpha), (gen, scores, deg) in arms.items()
        },
        "baseline_sequences": baseline_generated,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
