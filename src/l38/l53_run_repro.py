"""L53: activation-steering run for binding affinity, reusing L42's exact
difference-of-means construction, steering hook, degeneracy filter, and paired-
bootstrap significance test. Only the target dataset (ProteinGym
RASK_HUMAN_Weng_2022_binding-DARPin_K55) and the scoring proxy
(src/l38/l53_binding_affinity_steering.py) are new.

Judged against docs/L50_CAPABILITY_GAIN_PROTOCOL.md's 6 criteria:
  1 -- real direction vs matched-norm random direction, direct paired bootstrap.
  2 -- dose-response across ALPHAS, checked only over SAFE_ALPHAS (see below).
  3 -- residue-exclusion robustness at best_alpha.
  4 -- proxy pre-validated against real DMS labels BEFORE this script existed
       (r=+0.795 full / +0.797 held-out test, plus weight-shuffle and
       mutational-load controls -- see the proxy module's docstring).
  5 -- does not apply: binding affinity is a NEW property with no existing
       technique in this project to beat, same situation as L51's aggregation.
  6 -- N_EVAL_SEQS=150, per the protocol's n>=150 rule for a new-property claim.

One structural difference from every prior target in this arc worth stating
plainly: thermostability/aggregation/disorder each drew their high and low
groups from thousands of unrelated proteins, so the difference-of-means vector
separated genuinely different sequences. A ProteinGym DMS assay instead varies
ONE 188-residue KRAS backbone at 1-2 positions, so the low-binding and
high-binding groups here are ~99% identical in sequence. The steering vector is
therefore being asked to capture a far subtler contrast, and a null result would
be weak evidence about binding affinity in general as opposed to evidence about
this vector's ability to resolve near-identical sequences. Recorded here so the
verdict is read with the right scope, not discovered afterwards.
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
    paired_bootstrap_mean_diff,
)
from src.l38.l53_binding_affinity_steering import (
    binding_affinity_proxy,
    binding_affinity_proxy_excluding,
    mutational_sensitivity_weights,
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = (
    Path(__file__).resolve().parent
    / "data_cache"
    / "binding"
    / "RASK_HUMAN_Weng_2022_binding-DARPin_K55.parquet"
)
OUT_DIR = Path(__file__).resolve().parent / "l53_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42/L43/L51/L52's convention
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 150  # per L50 criterion 6's n>=150 rule for a new-property claim
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # reuse L42's empirically-derived range
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30

# L42's established constraint (docs/L42_STEERING_REPRO.md "Honest verdict",
# restated in docs/L52_LAYER_SUBSET_STEERING.md's "Critical correction"):
# alpha >= 1.0 degenerates this eval methodology (single-shot argmax mask-fill)
# into homopolymer collapse regardless of whether the steering vector is doing
# anything real, so any alpha >= 1.0 comparison is untrustworthy no matter what
# its bootstrap CI says. best_alpha and the dose-response check are both
# restricted to this window -- ranging outside it produced a real spurious PASS
# in L52's first draft, purely from two arms collapsing at different alphas.
SAFE_ALPHAS = (0.1, 0.25, 0.5)


class MultiLayerSteeringHook:
    """Identical to L42/L43/L51/L52's hook -- adds alpha*direction to a layer's
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


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_parquet(DATA_PATH)
    df = df[~df["is_indel"].astype(bool)]
    df = df[df["mutated_seq"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    reference_seq = df["target_seq"].iloc[0]
    print(f"usable (non-indel, length-filtered) variants: {len(df)}", flush=True)
    print(f"reference backbone length: {len(reference_seq)}", flush=True)

    # Disjoint vector-building / eval split. The proxy's sensitivity weights are
    # fit on the VECTOR split's labels only -- fitting them on the eval split
    # would leak eval labels into the scorer that judges the eval sequences.
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_vector = int(0.7 * len(shuffled))
    vector_pool = shuffled.iloc[:n_vector].reset_index(drop=True)
    eval_pool = shuffled.iloc[n_vector:].reset_index(drop=True)
    print(f"vector pool: {len(vector_pool)}, eval pool: {len(eval_pool)}", flush=True)

    weights = mutational_sensitivity_weights(
        vector_pool["mutant"].tolist(), vector_pool["DMS_score"].values, len(reference_seq)
    )
    print(f"sensitivity weights fit on vector pool only: "
          f"{int((weights > 0).sum())}/{len(reference_seq)} positions nonzero", flush=True)

    def score_binding(sequences):
        return np.array([binding_affinity_proxy(s, reference_seq, weights) for s in sequences])

    # high/low groups by REAL experimental binding score, percentile split on
    # the real label -- never on the proxy (that would be circular).
    labels = vector_pool["DMS_score"].astype(float).values
    low_threshold = np.percentile(labels, 20.0)
    high_threshold = np.percentile(labels, 80.0)
    low_group = vector_pool[vector_pool["DMS_score"].astype(float) <= low_threshold]["mutated_seq"].tolist()[
        :N_VECTOR_SEQS_PER_GROUP
    ]
    high_group = vector_pool[vector_pool["DMS_score"].astype(float) >= high_threshold]["mutated_seq"].tolist()[
        :N_VECTOR_SEQS_PER_GROUP
    ]
    print(f"vector-building groups: {len(low_group)} low-binding, {len(high_group)} high-binding", flush=True)
    assert len(low_group) >= N_VECTOR_SEQS_PER_GROUP, "not enough low-binding sequences"
    assert len(high_group) >= N_VECTOR_SEQS_PER_GROUP, "not enough high-binding sequences"

    # eval sequences: LOW-binding variants from the held-out eval pool --
    # steering should push them toward better binding, mirroring L42's low-Tm
    # and L51's low-resistance eval-sequence choice.
    eval_labels = eval_pool["DMS_score"].astype(float).values
    eval_low_threshold = np.percentile(eval_labels, 50.0)
    eval_sequences = eval_pool[eval_pool["DMS_score"].astype(float) <= eval_low_threshold][
        "mutated_seq"
    ].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences (low-binding, held out from vector construction): {len(eval_sequences)}", flush=True)
    assert len(eval_sequences) >= N_EVAL_SEQS, f"need {N_EVAL_SEQS} eval sequences, got {len(eval_sequences)}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-binding group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-binding group (all layers)...", flush=True)
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
        return generated, score_binding(generated)

    def score_arm(vectors, alpha):
        generated, scores = generate_then_score(vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    results = {"real_direction": {}, "random_control": {}}
    all_sequences = {"baseline": None, "real_direction": {}, "random_control": {}}

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, "
          f"degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)
    results["baseline"] = {
        "mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()),
        "n": len(baseline_scores), "n_degenerate": int(baseline_degenerate.sum()),
    }
    all_sequences["baseline"] = baseline_generated

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
        _, real_scores, real_degenerate = real_by_alpha[alpha]
        _, random_scores, random_degenerate = random_by_alpha[alpha]
        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())
        results["real_direction"][alpha] = {
            "mean": float(real_scores.mean()), "n_degenerate": int(real_degenerate.sum())
        }
        results["random_control"][alpha] = {
            "mean": float(random_scores.mean()), "n_degenerate": int(random_degenerate.sum())
        }
        if n_kept < MIN_NONDEGENERATE_PAIRS:
            real_vs_random_by_alpha[alpha] = {
                "point_estimate": None, "ci_lower": None, "ci_upper": None,
                "significant_at_95pct": False, "n": n_kept,
                "excluded_reason": f"only {n_kept} non-degenerate pairs, below MIN_NONDEGENERATE_PAIRS={MIN_NONDEGENERATE_PAIRS}",
            }
            print(f"\nalpha={alpha}: EXCLUDED -- only {n_kept} non-degenerate pairs", flush=True)
            continue
        bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
        bootstrap["pct_sequences_real_beats_random"] = float((real_scores[keep] > random_scores[keep]).mean())
        real_vs_random_by_alpha[alpha] = bootstrap
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] "
              f"sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={bootstrap['pct_sequences_real_beats_random']:.3f}", flush=True)

    # Criterion 2: dose-response, restricted to SAFE_ALPHAS (see SAFE_ALPHAS).
    valid_alphas = [a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a]["point_estimate"] is not None]
    dose_response_ok = (
        dose_response_is_monotonic_then_collapsing(
            valid_alphas, [real_vs_random_by_alpha[a]["point_estimate"] for a in valid_alphas]
        )
        if len(valid_alphas) >= 2
        else False
    )

    # Criterion 3: residue-exclusion robustness at the strongest SAFE alpha.
    best_alpha = max(
        (a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a].get("significant_at_95pct")),
        key=lambda a: real_vs_random_by_alpha[a]["point_estimate"],
        default=None,
    )
    robustness_check = None
    if best_alpha is not None:
        from collections import Counter

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
        real_excl = np.array([
            binding_affinity_proxy_excluding(s, reference_seq, weights, top_residues)
            for s in np.array(real_generated)[keep]
        ])
        random_excl = np.array([
            binding_affinity_proxy_excluding(s, reference_seq, weights, top_residues)
            for s in np.array(random_generated)[keep]
        ])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl, real_excl, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha, "excluded_residues": sorted(top_residues),
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): "
              f"diff={excl_bootstrap['point_estimate']:.4f} "
              f"[{excl_bootstrap['ci_lower']:.4f}, {excl_bootstrap['ci_upper']:.4f}] "
              f"sig={excl_bootstrap['significant_at_95pct']}", flush=True)

    crit1 = best_alpha is not None
    crit2 = dose_response_ok
    crit3 = robustness_check is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
    crit4 = True  # proxy pre-validated against real DMS labels, r=+0.795/+0.797
    crit6 = len(eval_sequences) >= 150
    criteria = {
        "1_beats_controls": crit1,
        "2_dose_response": crit2,
        "3_residue_robust": crit3,
        "4_proxy_pre_validated": crit4,
        "6_adequately_powered": crit6,
    }
    # Criterion 5 omitted by design: no existing technique for this property to
    # beat (same as L51). Operative bar is 1-4 + 6.
    n_pass = sum(criteria.values())
    if n_pass == 5:
        decision = "PASS"
    elif not crit1 or not crit3:
        decision = "KILL"
    else:
        decision = "AMBIGUOUS"

    verdict = {
        "criteria": criteria,
        "decision": decision,
        "best_alpha": best_alpha,
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "robustness_check": robustness_check,
        "proxy_validation": {
            "note": "pre-validated before this run; see l53_binding_affinity_steering.py docstring",
            "pearson_r_full": 0.795,
            "pearson_r_heldout_test": 0.797,
            "weight_shuffle_control_r": -0.001,
        },
    }

    print("\n=== L53 VERDICT (binding affinity, degenerate-filtered, paired-bootstrapped, "
          "residue-exclusion-checked) ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results["verdict"] = verdict
    results["dataset"] = {
        "path": str(DATA_PATH), "n_usable_variants": len(df),
        "reference_length": len(reference_seq),
        "n_vector_pool": len(vector_pool), "n_eval_pool": len(eval_pool),
    }
    results["raw_scores"] = {
        f"real__{alpha}": real_by_alpha[alpha][1].tolist() for alpha in ALPHAS
    }
    results["raw_scores"].update({
        f"random__{alpha}": random_by_alpha[alpha][1].tolist() for alpha in ALPHAS
    })
    results["raw_scores"]["baseline"] = baseline_scores.tolist()
    results["raw_sequences"] = {
        "baseline": baseline_generated,
        **{f"real__{alpha}": real_by_alpha[alpha][0] for alpha in ALPHAS},
        **{f"random__{alpha}": random_by_alpha[alpha][0] for alpha in ALPHAS},
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
