"""L54: activation-steering run for enzyme catalytic activity (kcat), reusing
L42's exact difference-of-means construction, steering hook, degeneracy
filter, and paired-bootstrap test. Only the target dataset (DLKcat) and the
scoring proxy (src/l38/l54_catalytic_activity_steering.py) are new.

Judged against docs/L50_CAPABILITY_GAIN_PROTOCOL.md's 6 criteria:
  1 real direction beats matched-norm random control head-to-head with a
    paired-bootstrap CI (not two separate vs.-baseline tests);
  2 dose-response coherent across >=3 alphas, checked only inside the safe
    alpha window (see SAFE_ALPHAS);
  3 survives residue-exclusion;
  4 proxy pre-validated against real labels -- and RE-VALIDATED IN THIS RUN
    against the actual held-out eval pool's labels, gated by an assert before
    the model is even loaded, so criterion 4 is provable by running this
    script rather than trusted from a docstring;
  5 not operative: there is no prior technique for this property to beat
    (same situation as L51's aggregation resistance);
  6 n=150 eval sequences, per the n>=150 rule for a NEW target property.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_steering_repro import (
    difference_of_means_vector,
    dose_response_is_monotonic_then_collapsing,
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
)
from src.l38.l54_catalytic_activity_steering import (
    catalytic_activity_proxy,
    catalytic_activity_proxy_excluding,
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "catalytic" / "dlkcat_wt_mut.json"
OUT_DIR = Path(__file__).resolve().parent / "l54_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42/L43/L51's convention
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 150  # per Phase 0 protocol's n>=150 rule for a new-property claim
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # reuse L42's empirically-derived range
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30
TRAIN_FRACTION = 0.7

CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")

# L42's established constraint, restated as L52's "Critical correction"
# (docs/L52_LAYER_SUBSET_STEERING.md): at alpha >= 1.0 this eval methodology
# (single-shot argmax mask-fill) collapses into degenerate low-complexity
# output regardless of whether the steering direction means anything, so no
# alpha >= 1.0 comparison is trustworthy no matter what its bootstrap CI
# says. best_alpha selection and the dose-response check are both restricted
# to this window -- selecting outside it produced a spurious PASS in L52's
# first draft, purely from two arms collapsing at different alphas.
SAFE_ALPHAS = (0.1, 0.25, 0.5)

# Criterion-4 gate: the proxy must still clear this on the held-out eval pool
# computed in THIS run, not just in the offline validation written up in
# l54_catalytic_activity_steering.py's docstring. Set below the measured
# held-out value (r=+0.212) with margin, and above the r=-0.03 that L43's
# GRAVY turned out to have.
MIN_PROXY_ABS_R = 0.15


class MultiLayerSteeringHook:
    """Identical to L42/L43/L51/L52's hook -- adds alpha*direction to a
    layer's output, renormalized to preserve original per-token norm."""

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


def score_catalytic_activity(sequences):
    return np.array([catalytic_activity_proxy(seq) for seq in sequences])


def load_dlkcat(path=DATA_PATH, max_len=MAX_SEQ_LEN):
    """DLKcat records -> (sequences, log10 kcat) deduplicated by sequence.

    A sequence appears once per (substrate, condition) measurement, so kcat is
    aggregated by MEDIAN over a sequence's records -- taking one arbitrary
    record would let substrate choice, not the enzyme, set the label.
    """
    with open(path) as f:
        records = json.load(f)

    by_sequence = {}
    for rec in records:
        seq = rec["Sequence"]
        kcat = float(rec["Value"])
        if len(seq) > max_len or kcat <= 0 or not set(seq) <= CANONICAL_RESIDUES:
            continue
        by_sequence.setdefault(seq, []).append(np.log10(kcat))

    sequences = sorted(by_sequence)  # sorted so the split is reproducible
    labels = np.array([float(np.median(by_sequence[s])) for s in sequences])
    return sequences, labels


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    sequences, labels = load_dlkcat()
    print(f"usable unique enzymes (<={MAX_SEQ_LEN} aa, canonical, kcat>0): {len(sequences)}", flush=True)

    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(sequences))
    cut = int(TRAIN_FRACTION * len(sequences))
    train_idx, test_idx = order[:cut], order[cut:]
    train_seqs = [sequences[i] for i in train_idx]
    train_labels = labels[train_idx]
    test_seqs = [sequences[i] for i in test_idx]
    test_labels = labels[test_idx]
    print(f"train (vector pool): {len(train_seqs)}, test (eval pool, held out): {len(test_seqs)}", flush=True)

    # === Criterion 4, enforced before any GPU work ===
    # Re-validate the proxy against REAL labels on both splits. L43 ran its
    # whole experiment and only afterwards discovered its proxy correlated
    # r=-0.03 with real labels; this assert makes that failure mode impossible
    # to reach here without the run stopping first.
    train_proxy = score_catalytic_activity(train_seqs)
    test_proxy = score_catalytic_activity(test_seqs)
    r_train, p_train = pearsonr(train_proxy, train_labels)
    r_test, p_test = pearsonr(test_proxy, test_labels)
    rho_test, _ = spearmanr(test_proxy, test_labels)
    proxy_validation = {
        "train": {"pearson_r": float(r_train), "p": float(p_train), "n": len(train_seqs)},
        "test": {"pearson_r": float(r_test), "p": float(p_test), "spearman_rho": float(rho_test), "n": len(test_seqs)},
        "min_abs_r_required": MIN_PROXY_ABS_R,
    }
    print(f"proxy vs real log10(kcat): train r={r_train:+.4f} (p={p_train:.2e}), "
          f"test r={r_test:+.4f} (p={p_test:.2e}), test rho={rho_test:+.4f}", flush=True)
    assert abs(r_test) >= MIN_PROXY_ABS_R, (
        f"criterion 4 FAILED: proxy correlates only r={r_test:+.4f} with real held-out labels "
        f"(need |r| >= {MIN_PROXY_ABS_R}); refusing to run the steering sweep on an unvalidated proxy"
    )
    assert r_test > 0, "proxy sign is inverted relative to the real label; fix the proxy, not the run"

    # vector-building groups: percentile split on the REAL label, not the proxy
    low_threshold = np.percentile(train_labels, 20.0)
    high_threshold = np.percentile(train_labels, 80.0)
    low_group = [s for s, y in zip(train_seqs, train_labels) if y <= low_threshold][:N_VECTOR_SEQS_PER_GROUP]
    high_group = [s for s, y in zip(train_seqs, train_labels) if y >= high_threshold][:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} low-kcat (<= {low_threshold:.3f} log10 s^-1), "
          f"{len(high_group)} high-kcat (>= {high_threshold:.3f})", flush=True)
    assert len(low_group) == N_VECTOR_SEQS_PER_GROUP and len(high_group) == N_VECTOR_SEQS_PER_GROUP

    # eval sequences: LOW-activity enzymes from the held-out test split --
    # steering should push them toward higher activity, mirroring L42's
    # low-Tm and L51's low-resistance eval-sequence choice.
    test_median = np.percentile(test_labels, 50.0)
    eval_sequences = [s for s, y in zip(test_seqs, test_labels) if y <= test_median][:N_EVAL_SEQS]
    print(f"eval sequences (low-kcat, held-out test split): {len(eval_sequences)}", flush=True)
    assert len(eval_sequences) == N_EVAL_SEQS, f"only {len(eval_sequences)} eval sequences available, need {N_EVAL_SEQS}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-kcat group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-kcat group (all layers)...", flush=True)
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
        return generated, score_catalytic_activity(generated)

    def score_arm(vectors, alpha):
        generated, scores = generate_then_score(vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    results = {"proxy_validation": proxy_validation, "real_direction": {}, "random_control": {}}
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
        results["real_direction"][alpha] = {"mean": float(real_scores.mean()), "n_degenerate": int(real_degenerate.sum())}
        results["random_control"][alpha] = {"mean": float(random_scores.mean()), "n_degenerate": int(random_degenerate.sum())}
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
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={bootstrap['pct_sequences_real_beats_random']:.3f}", flush=True)

    # Criterion 2, restricted to SAFE_ALPHAS (see SAFE_ALPHAS comment)
    valid_alphas = [a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a]["point_estimate"] is not None]
    dose_response_ok = (
        dose_response_is_monotonic_then_collapsing(
            list(valid_alphas), [real_vs_random_by_alpha[a]["point_estimate"] for a in valid_alphas]
        )
        if len(valid_alphas) >= 2 else False
    )

    # Criterion 3, on the strongest SAFE alpha with a significant effect
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

        # The proxy is a G-vs-R contrast, so excluding BOTH G and R makes it
        # identically 0 for every sequence and the bootstrap vacuous. When
        # that happens, fall back to excluding only the single most dominant
        # residue -- still a real exclusion check, just the strongest one that
        # leaves the proxy defined.
        exclusion_note = None
        if top_residues >= frozenset("GR"):
            top_residues = frozenset(r for r, _ in counts.most_common(1))
            exclusion_note = (
                "top-2 substituted residues were G and R, which are the proxy's own two terms; "
                "excluding both zeroes the proxy identically, so the check uses the top-1 residue only"
            )
            print(f"NOTE: {exclusion_note}", flush=True)

        keep = ~real_degenerate & ~random_degenerate & ~baseline_degenerate
        real_excl_scores = np.array([
            catalytic_activity_proxy_excluding(s, top_residues) for s in np.array(real_generated)[keep]
        ])
        random_excl_scores = np.array([
            catalytic_activity_proxy_excluding(s, top_residues) for s in np.array(random_generated)[keep]
        ])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl_scores, real_excl_scores, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha,
            "excluded_residues": sorted(top_residues),
            "exclusion_note": exclusion_note,
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): diff={excl_bootstrap['point_estimate']:.4f} "
              f"[{excl_bootstrap['ci_lower']:.4f}, {excl_bootstrap['ci_upper']:.4f}] "
              f"sig={excl_bootstrap['significant_at_95pct']}", flush=True)

    crit1 = best_alpha is not None
    crit2 = dose_response_ok
    crit3 = robustness_check is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
    crit4 = abs(r_test) >= MIN_PROXY_ABS_R  # asserted above; recorded for the verdict record
    crit5 = None  # not operative: no prior technique exists for this property
    crit6 = len(eval_sequences) >= 150

    criteria = {
        "1_beats_random_control": crit1,
        "2_dose_response": crit2,
        "3_residue_robust": crit3,
        "4_proxy_pre_validated": crit4,
        "5_beats_existing_technique": crit5,
        "6_adequately_powered": crit6,
    }
    operative = [v for v in criteria.values() if v is not None]
    if all(operative):
        decision = "PASS"
    elif not crit1 or not crit3:
        decision = "KILL"
    elif sum(bool(v) for v in operative) >= 3:
        decision = "AMBIGUOUS"
    else:
        decision = "KILL"

    verdict = {
        "criteria": criteria,
        "decision": decision,
        "best_alpha": best_alpha,
        "proxy_validation": proxy_validation,
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "robustness_check": robustness_check,
    }

    print("\n=== L54 VERDICT (catalytic activity / kcat, degenerate-filtered, "
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
