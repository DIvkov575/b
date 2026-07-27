"""L42: reproduce Huang et al.'s activation steering (arXiv:2509.07983) on
ESM2-650M toward thermostability, using difference-of-means vectors (not a
raw SAE feature -- the fix flagged by the L41 post-mortem literature check).

Sanity-check run BEFORE any new steering claim: if this doesn't reproduce a
clear steering effect, the harness (not the technique) has a bug, and no
new-target result from the same pipeline should be trusted until it's found.
See docs/L42_STEERING_REPRO.md for the full protocol and PASS/KILL rule.
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
    ivywrel_fraction,
    paired_bootstrap_mean_diff,
    split_by_percentile,
)

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "meltome" / "mixed_split.csv"
OUT_DIR = Path(__file__).resolve().parent / "l42_repro_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 400  # keep forward passes cheap; matches L41's convention
N_VECTOR_SEQS_PER_GROUP = 150  # sequences used to BUILD the steering vector
N_EVAL_SEQS = 60  # held-out sequences steered/scored, disjoint from vector-building set
# Alpha range re-derived empirically (not inherited from L41's kinase run):
# a manual sweep at mask_frac=0.3 showed alpha=2-16 saturates the model into
# degenerate poly-leucine output even at alpha=2 -- identical generated
# sequences (and identical scores to 15+ decimal places) across 2/4/8/16
# confirmed this. The graded, non-degenerate regime is roughly alpha in
# [0.02, 1.0]; alpha=2.0 kept as an intentional "does it eventually collapse"
# upper anchor, not the main operating range.
ALPHAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SEED = 0
N_BOOT = 10000


class MultiLayerSteeringHook:
    """Adds alpha*direction[layer] to a specific transformer layer's output,
    renormalized to preserve the original per-token activation norm -- per
    Huang et al.'s method. One hook instance per layer; register on all
    layers except the embedding layer (Huang et al.'s described scope)."""

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
    for every transformer layer (excluding the embedding layer), matching
    Huang et al.'s per-layer difference-of-means construction."""
    n_layers = model.config.num_hidden_layers
    per_layer_activations = {layer: [] for layer in range(n_layers)}

    for seq in sequences:
        seq = seq[:max_len]
        enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states[0] = embedding output; hidden_states[i+1] = output of layer i
        for layer in range(n_layers):
            hidden = out.hidden_states[layer + 1].squeeze(0).float()
            per_layer_activations[layer].append(hidden.mean(dim=0).cpu().numpy())

    return {layer: np.stack(vals, axis=0) for layer, vals in per_layer_activations.items()}


MASK_FRACTION = 0.3  # NOT L41's 0.8 -- checked empirically first this time:
# at mask_frac=0.8, even the UNSTEERED (alpha=0) baseline generates degenerate
# output (X tokens, runaway L/W runs) because single-shot infilling of 80% of
# a sequence from one forward pass is simply too hard for this model, independent
# of any steering. 0.3 keeps baseline generation coherent while still giving
# steering enough masked positions to visibly act on (confirmed via manual
# alpha sweep: graded, non-degenerate effect through ~alpha=0.5-1.0, collapse
# starting around alpha=1.0-2.0 -- not a flat "no effect at any usable alpha"
# situation like L41's mistaken 0.8/alpha-2-16 combination produced).


@torch.no_grad()
def mask_fill_generate(model, tokenizer, sequence, mask_fraction, seed, device, max_len=MAX_SEQ_LEN):
    """Mask most of the sequence, single-shot-predict the masked positions
    (argmax per position), return the generated sequence. This is what gets
    STEERED (the hook is active during this forward pass) -- Huang et al.'s
    actual causal test is on GENERATED output, not on the likelihood of an
    unmodified input under a perturbed model (an earlier draft of this script
    conflated the two -- caught in a smoke test before the real run, see
    docs/L42_STEERING_REPRO.md)."""
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
    """Independent thermostability PROXY: IVYWREL fraction (Zeldovich et al.
    2007, Kreil & Ouzounis 2001) -- higher = more thermostable-like
    composition, per comparative thermophile/mesophile proteome genomics.

    Replaces TWO earlier, both-confounded proxies (documented in
    docs/L42_STEERING_REPRO.md): (1) self-likelihood, confounded because
    unusual-looking (not necessarily less stable) output scores as "bad";
    (2) the Guruprasad instability index, confounded because it happens to
    score this steering vector's dominant failure mode (poly-leucine
    collapse) as artificially stable. IVYWREL was checked to NOT be "leucine
    in disguise" -- the effect survives with leucine excluded from the
    residue set entirely (see docs/L42_STEERING_REPRO.md RESULTS v2).
    """
    return np.array([ivywrel_fraction(seq) for seq in sequences])


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    print(f"usable (length-filtered) sequences: {len(df)}", flush=True)

    rng = np.random.RandomState(SEED)
    shuffled = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    vector_pool = shuffled.iloc[: 2 * N_VECTOR_SEQS_PER_GROUP + 500]
    eval_pool = shuffled.iloc[2 * N_VECTOR_SEQS_PER_GROUP + 500 :]

    low_group, high_group = split_by_percentile(
        vector_pool["sequence"].tolist(), vector_pool["label"].values, low_pct=20.0, high_pct=80.0
    )
    low_group = low_group[:N_VECTOR_SEQS_PER_GROUP]
    high_group = high_group[:N_VECTOR_SEQS_PER_GROUP]
    print(f"vector-building groups: {len(low_group)} low-Tm, {len(high_group)} high-Tm", flush=True)

    # Eval sequences: take from the LOW end of the eval pool (steering should
    # push a low-stability sequence toward higher-stability-like activations),
    # disjoint from the vector-building pool by construction (different slice).
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
    # Match random-direction norm per layer to the real steering vector's norm,
    # so any difference in effect isn't just explained by a magnitude mismatch.
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
        """Generate (with hooks active if alpha != 0) then remove hooks
        BEFORE scoring -- scoring must always run on the unperturbed model,
        per the fix above."""
        generated = []
        handles = apply_hooks(vectors, alpha) if alpha != 0.0 else []
        try:
            for i, seq in enumerate(eval_sequences):
                generated.append(mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED + i, device))
        finally:
            remove_hooks(handles)
        scores = score_thermostability_proxy(generated)
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

    MIN_NONDEGENERATE_PAIRS = 30  # below this, a bootstrap CI is too noisy to
    # trust regardless of what it says -- this is exactly what went wrong when
    # alpha=1.0 first looked like a clean PASS on only 5/60 surviving pairs
    # (see docs/L42_STEERING_REPRO.md RESULTS v1): a handful of sequences that
    # happened to sit just under the degeneracy cutoff still carried a milder
    # version of the same collapse artifact.

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
        # Pair on indices where NEITHER arm NOR the baseline collapsed at this
        # alpha -- a collapsed baseline or collapsed random-control generation
        # makes the comparison meaningless at that index too, not just a
        # collapsed real-direction generation.
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
        # PASS requires the real direction to significantly beat the random
        # control HEAD-TO-HEAD (direct paired bootstrap, not two separate
        # vs.-baseline tests) at at least one alpha with enough surviving
        # non-degenerate pairs to trust the CI.
        "decision": "PASS" if any(v["significant_at_95pct"] for v in real_vs_random_by_alpha.values()) else "INCONCLUSIVE",
    }

    print("\n=== L42 VERDICT (degenerate-filtered, paired-bootstrapped) ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)

    results["verdict"] = verdict
    results["all_sequences"] = all_sequences
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
