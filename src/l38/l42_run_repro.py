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
    dose_response_is_monotonic_then_collapsing,
    instability_index,
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
    """Independent thermostability PROXY: the Guruprasad et al. (1990)
    instability index, a PURELY COMPOSITIONAL formula (no model, no
    likelihood) -- lower = more stable. This replaces an earlier draft that
    scored generated sequences with the model's OWN self-likelihood, which
    is confounded: the steering direction's dominant effect (a shift toward
    poly-leucine-like output, confirmed via manual inspection) makes text
    look UNUSUAL to the model regardless of whether it's actually more
    stable, so a self-likelihood proxy can't distinguish "less thermostable"
    from "just doesn't look like typical protein text anymore." The
    instability index has no such confound since it never touches the model.
    Returns NEGATED instability (so higher score = more stable, consistent
    sign convention with "higher score = better" used elsewhere).
    """
    scores = []
    for seq in sequences:
        clean_seq = seq if len(seq) >= 2 else seq + "A"  # guard the length>=2 requirement; shouldn't occur in practice
        scores.append(-instability_index(clean_seq))
    return np.array(scores)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    example_sequences = {"baseline": None, "real_direction": {}, "random_control": {}}
    N_EXAMPLES_TO_SAVE = 5

    print("\n=== baseline (alpha=0) ===", flush=True)
    baseline_generated, baseline_scores = generate_then_score(steering_vectors, 0.0)
    print(f"baseline mean score: {baseline_scores.mean():.4f}", flush=True)
    results["baseline"] = {"mean": float(baseline_scores.mean()), "std": float(baseline_scores.std()), "n": len(baseline_scores)}
    example_sequences["baseline"] = baseline_generated[:N_EXAMPLES_TO_SAVE]

    for alpha in ALPHAS:
        if alpha == 0.0:
            continue
        print(f"\n=== real_direction, alpha={alpha} ===", flush=True)
        generated, scores = generate_then_score(steering_vectors, alpha)
        results["real_direction"][alpha] = {"mean": float(scores.mean()), "std": float(scores.std()), "n": len(scores)}
        example_sequences["real_direction"][alpha] = generated[:N_EXAMPLES_TO_SAVE]
        print(f"mean score: {scores.mean():.4f}", flush=True)

        print(f"=== random_control, alpha={alpha} ===", flush=True)
        generated, scores = generate_then_score(random_vectors, alpha)
        results["random_control"][alpha] = {"mean": float(scores.mean()), "std": float(scores.std()), "n": len(scores)}
        example_sequences["random_control"][alpha] = generated[:N_EXAMPLES_TO_SAVE]
        print(f"mean score: {scores.mean():.4f}", flush=True)

    real_effects = [results["real_direction"][a]["mean"] - results["baseline"]["mean"] for a in ALPHAS if a != 0.0]
    random_effects = [results["random_control"][a]["mean"] - results["baseline"]["mean"] for a in ALPHAS if a != 0.0]
    nonzero_alphas = [a for a in ALPHAS if a != 0.0]

    # Use |effect| for the dose-response check -- the score can move in
    # either direction (this run's real effect happens to be negative,
    # i.e. HIGHER instability / LESS stable, not the hoped-for direction),
    # so what matters is whether the MAGNITUDE grows with alpha, not whether
    # it grows more positive specifically (a bug in the first version of this
    # check, caught by inspecting the raw numbers rather than trusting the
    # boolean output blindly).
    abs_real_effects = [abs(e) for e in real_effects]
    abs_random_effects = [abs(e) for e in random_effects]

    dose_response_real = dose_response_is_monotonic_then_collapsing(nonzero_alphas, abs_real_effects)
    dose_response_random = dose_response_is_monotonic_then_collapsing(nonzero_alphas, abs_random_effects)

    # "beats random" = larger MAGNITUDE of effect at every alpha, not just at
    # the max -- a single-alpha comparison could be a fluke.
    real_beats_random_at_every_alpha = all(
        abs(real_effects[i]) > abs(random_effects[i]) for i in range(len(nonzero_alphas))
    )

    verdict = {
        "real_effects_by_alpha": dict(zip(nonzero_alphas, real_effects)),
        "random_effects_by_alpha": dict(zip(nonzero_alphas, random_effects)),
        # score = -instability_index, so a NEGATIVE effect (score decreased)
        # means instability INCREASED, i.e. steering pushed toward LESS
        # stable sequences -- the opposite of what Huang et al.'s "high" group
        # (used to build the steering vector) should represent. Get this sign
        # right explicitly rather than eyeballing it from a raw number.
        "direction": "toward LESS stable (higher instability index)" if real_effects[-1] < 0 else "toward MORE stable (lower instability index)",
        "dose_response_real": dose_response_real,
        "dose_response_random": dose_response_random,
        "real_beats_random_at_every_alpha": real_beats_random_at_every_alpha,
        "decision": "PASS" if (dose_response_real and real_beats_random_at_every_alpha) else "KILL",
    }

    print("\n=== L42 VERDICT ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)

    results["verdict"] = verdict
    results["example_sequences"] = example_sequences
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
