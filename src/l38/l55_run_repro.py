"""L55: activation-steering run for intrinsic disorder (IDR) content,
reusing L42's exact difference-of-means construction, degeneracy filter, and
paired-bootstrap significance test. Only the target dataset (DisProt current
release, cleaned) and scoring proxy (TOP-IDP based,
src/l38/l55_disorder_steering.py) are new.

Judged against docs/L50_CAPABILITY_GAIN_PROTOCOL.md's 6 criteria:
  1 -- real direction beats matched-norm random control head-to-head, paired
       bootstrap, per alpha.
  2 -- dose-response across SAFE_ALPHAS (see below), not one lucky alpha.
  3 -- survives residue-exclusion robustness check.
  4 -- proxy pre-validated against REAL per-residue DisProt labels BEFORE
       this run: r=+0.449 whole-sequence (test-split +0.482, partial r=+0.428
       controlling length), per-residue AUC=0.713 over 354k residues. See
       src/l38/l55_disorder_steering.py's docstring for the full table of
       four candidate proxies, all validated before one was chosen.
  5 -- N/A: disorder is a NEW target property with no existing technique in
       this project to beat head-to-head (same reasoning as L51). Criteria
       1-4 + 6 are the operative PASS bar.
  6 -- n=150 eval sequences, per the protocol's n>=150 rule for a new
       target property.

Direction convention: steering pushes ORDERED sequences TOWARD disorder, so
the difference-of-means vector runs low-disorder -> high-disorder and eval
sequences are drawn from the ordered (low-disorder) half of a held-out pool.
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
from src.l38.l55_disorder_steering import disorder_proxy, disorder_proxy_excluding

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "disorder" / "disprot_clean.csv"
OUT_DIR = Path(__file__).resolve().parent / "l55_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # matches L42/L43/L51/L52's convention
N_VECTOR_SEQS_PER_GROUP = 150
N_EVAL_SEQS = 150  # per L50 criterion 6's n>=150 rule for a new-property claim
ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]  # reuse L42's empirically-derived range
MASK_FRACTION = 0.3
SEED = 0
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30

# Target-specific caveat: `is_degenerate_sequence` (>25% single AA) penalizes
# low-complexity sequence, which real IDRs genuinely are -- measured on this
# dataset it flags 1.5% of ordered but 6.7% of fully-disordered REAL DisProt
# sequences. Mild, so the filter is kept unchanged for comparability with
# L42/L51/L52, but it biases slightly AGAINST detecting a real disorder
# effect (conservative), and a near-miss result should be re-checked with a
# looser threshold before being called a KILL.

# Size of the pool reserved for vector construction; the remainder is the
# disjoint held-out pool eval sequences are drawn from. 1000 of the 1615
# length-filtered sequences leaves 615 held out, of which the ordered half
# (308) comfortably covers N_EVAL_SEQS=150.
VECTOR_POOL_SIZE = 1000

# L42's own established constraint (docs/L42_STEERING_REPRO.md "Honest
# verdict"; docs/L52_LAYER_SUBSET_STEERING.md "Critical correction"):
# alpha >= 1.0 degenerates this single-shot argmax mask-fill eval into
# compositional collapse regardless of whether the steering vector is doing
# anything real, so any alpha >= 1.0 result is untrustworthy no matter what
# its bootstrap CI says. L52's first draft picked alpha=2.0 and got a
# spurious PASS purely from two arms collapsing at different alphas.
# best_alpha selection and the dose-response check are both restricted here.
SAFE_ALPHAS = (0.1, 0.25, 0.5)


class MultiLayerSteeringHook:
    """Identical to L42/L43/L51/L52's hook -- adds alpha*direction to a
    layer's output, renormalized to preserve original per-token activation
    norm."""

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


def score_disorder(sequences):
    return np.array([disorder_proxy(seq) for seq in sequences])


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    # Vector-building pool and eval pool are disjoint slices of one shuffle,
    # so eval sequences are genuinely held out from vector construction
    # (mirrors L52; DisProt ships no train/test split of its own, unlike
    # L51's aggregation dataset).
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    vector_pool = shuffled.iloc[:VECTOR_POOL_SIZE]
    eval_pool = shuffled.iloc[VECTOR_POOL_SIZE:]
    print(f"vector pool: {len(vector_pool)}, held-out eval pool: {len(eval_pool)}", flush=True)

    # high/low groups by the REAL per-residue-derived disorder fraction,
    # percentile split on the real label, not the proxy.
    labels = vector_pool["disorder_fraction"].astype(float).values
    low_threshold = np.percentile(labels, 20.0)
    high_threshold = np.percentile(labels, 80.0)
    low_group = vector_pool[vector_pool["disorder_fraction"].astype(float) <= low_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    high_group = vector_pool[vector_pool["disorder_fraction"].astype(float) >= high_threshold]["sequence"].tolist()[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} ordered (low-disorder), {len(high_group)} disordered (high-disorder)", flush=True)
    assert len(low_group) == N_VECTOR_SEQS_PER_GROUP, f"low group short: {len(low_group)}"
    assert len(high_group) == N_VECTOR_SEQS_PER_GROUP, f"high group short: {len(high_group)}"

    # eval sequences: ORDERED (low-disorder) sequences from the held-out
    # pool -- steering should push them TOWARD disorder, mirroring L42's
    # low-Tm eval-sequence choice.
    eval_labels = eval_pool["disorder_fraction"].astype(float).values
    eval_threshold = np.percentile(eval_labels, 50.0)
    eval_df = eval_pool[eval_pool["disorder_fraction"].astype(float) <= eval_threshold]
    eval_sequences = eval_df["sequence"].tolist()[:N_EVAL_SEQS]
    print(f"eval sequences (ordered, held out from vector construction): {len(eval_sequences)}, "
          f"mean real disorder_fraction: {eval_df['disorder_fraction'].head(N_EVAL_SEQS).mean():.4f}", flush=True)
    assert len(eval_sequences) >= MIN_NONDEGENERATE_PAIRS, "not enough eval sequences in held-out pool"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding ordered (low-disorder) group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding disordered (high-disorder) group (all layers)...", flush=True)
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
        return generated, score_disorder(generated)

    def score_arm(vectors, alpha):
        generated, scores = generate_then_score(vectors, alpha)
        degenerate = np.array([is_degenerate_sequence(s) for s in generated])
        return generated, scores, degenerate

    results = {"real_direction": {}, "random_control": {}}
    all_sequences = {"baseline": None, "real_direction": {}, "random_control": {}}

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean score: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)
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
            print(f"\nalpha={alpha}: EXCLUDED, only {n_kept} non-degenerate pairs", flush=True)
            continue
        bootstrap = paired_bootstrap_mean_diff(random_scores[keep], real_scores[keep], n_boot=N_BOOT, seed=SEED)
        bootstrap["pct_sequences_real_beats_random"] = float((real_scores[keep] > random_scores[keep]).mean())
        real_vs_random_by_alpha[alpha] = bootstrap
        print(f"\nalpha={alpha}: real-vs-random (n={n_kept}) diff={bootstrap['point_estimate']:.4f} "
              f"[{bootstrap['ci_lower']:.4f}, {bootstrap['ci_upper']:.4f}] sig={bootstrap['significant_at_95pct']} "
              f"pct_real_beats_random={bootstrap['pct_sequences_real_beats_random']:.3f}", flush=True)

    # Criterion 2: dose-response restricted to SAFE_ALPHAS -- alpha >= 1.0
    # is excluded by design (see SAFE_ALPHAS comment).
    valid_alphas = [a for a in SAFE_ALPHAS if real_vs_random_by_alpha[a]["point_estimate"] is not None]
    dose_response_ok = dose_response_is_monotonic_then_collapsing(
        valid_alphas, [real_vs_random_by_alpha[a]["point_estimate"] for a in valid_alphas]
    ) if len(valid_alphas) >= 2 else False

    # Criterion 3: residue-exclusion robustness at the strongest SAFE alpha.
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
        real_excl_scores = np.array([
            disorder_proxy_excluding(s, top_residues) for s in np.array(real_generated)[keep]
        ])
        random_excl_scores = np.array([
            disorder_proxy_excluding(s, top_residues) for s in np.array(random_generated)[keep]
        ])
        excl_bootstrap = paired_bootstrap_mean_diff(random_excl_scores, real_excl_scores, n_boot=N_BOOT, seed=SEED)
        robustness_check = {
            "alpha": best_alpha, "excluded_residues": sorted(top_residues),
            "diff_with_exclusion": excl_bootstrap,
        }
        print(f"robustness check (excluding {sorted(top_residues)}): diff={excl_bootstrap['point_estimate']:.4f} "
              f"[{excl_bootstrap['ci_lower']:.4f}, {excl_bootstrap['ci_upper']:.4f}] sig={excl_bootstrap['significant_at_95pct']}", flush=True)

    crit1 = best_alpha is not None
    crit2 = dose_response_ok
    crit3 = robustness_check is not None and robustness_check["diff_with_exclusion"]["significant_at_95pct"]
    crit4 = True  # TOP-IDP pre-validated vs real DisProt labels; see l55_disorder_steering.py
    crit5 = None  # N/A: new property, no existing technique to beat (as in L51)
    crit6 = len(eval_sequences) >= 150

    criteria = {"1_beats_controls": crit1, "2_dose_response": crit2, "3_residue_robust": crit3,
                "4_proxy_pre_validated": crit4, "5_beats_existing": crit5, "6_adequately_powered": crit6}
    operative = [crit1, crit2, crit3, crit4, crit6]
    if all(operative):
        decision = "PASS"
    elif not crit1 or not crit3:
        decision = "KILL"
    elif sum(bool(c) for c in operative) >= 3:
        decision = "AMBIGUOUS"
    else:
        decision = "KILL"

    verdict = {
        "criteria": criteria,
        "decision": decision,
        "best_alpha": best_alpha,
        "real_vs_random_by_alpha": real_vs_random_by_alpha,
        "robustness_check": robustness_check,
        "proxy_validation": {
            "proxy": "TOP-IDP (Campen et al. 2008), mean-pooled",
            "dataset": "DisProt current release, length<=400, canonical AA only, n=1615",
            "pearson_r_full": 0.449, "spearman_r_full": 0.362,
            "pearson_r_train": 0.440, "pearson_r_test": 0.482,
            "partial_r_controlling_length": 0.428,
            "per_residue_auc_window21": 0.713,
            "validated_before_run": True,
        },
    }

    print("\n=== L55 VERDICT (disorder steering; degenerate-filtered, paired-bootstrapped, residue-exclusion-checked) ===", flush=True)
    print(json.dumps(verdict, indent=2, default=str), flush=True)

    results["verdict"] = verdict
    results["raw_scores"] = {
        f"real__{alpha}": real_by_alpha[alpha][1].tolist() for alpha in ALPHAS
    }
    results["raw_scores"].update({
        f"random__{alpha}": random_by_alpha[alpha][1].tolist() for alpha in ALPHAS
    })
    results["raw_scores"]["baseline"] = baseline_scores.tolist()
    results["raw_sequences"] = {
        "baseline": baseline_generated,
        "real_direction": {str(a): all_sequences["real_direction"][a] for a in ALPHAS},
        "random_control": {str(a): all_sequences["random_control"][a] for a in ALPHAS},
    }
    results["eval_sequences"] = eval_sequences
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
