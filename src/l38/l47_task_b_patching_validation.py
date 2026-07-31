"""L47 Task B: validate the activation-patching harness (l47_activation_patching.py)
against L42's already-trusted thermostability result, BEFORE trusting it on
the novel Task A (redo Vig et al.'s attention-head finding causally).

Method, distinct from L42/L45's steering (ADD alpha*vector to activations):
this SUBSTITUTES the mean-pooled high-Tm-group activation directly in place
of a low-Tm eval sequence's own activation, at a given layer's residual
stream, restricted to the MASKED positions only (the ones actually being
predicted) -- broadcasting the same vector to every position, including
unmasked context the model needs to make any coherent prediction, was
tried first and found too destructive: a smoke test at N=5 showed 100%
degenerate output at every single layer (see docs/L47_ACTIVATION_PATCHING.md
for the full account). Patching only the positions under prediction is both
less destructive and more principled: it substitutes exactly the
information the model is being asked to fill in, not the context it reads
FROM. Real proteins have no natural token-to-token alignment across
different sequences (unlike paired-template LLM patching, e.g. "John gave
it to Mary" vs. "Mary gave it to John"), so this uses a single mean-pooled
vector broadcast across whichever positions get masked for a given eval
sequence, rather than any position-specific correspondence.

If a low-Tm sequence generated under this patch scores as more
thermostability-like than its own unpatched baseline, that's a genuinely
different causal test than L42's steering (substitution vs. addition)
converging on the same conclusion -- not a restatement of L42.

Reuses L42's exact data groups, degeneracy filter, IVYWREL scorer, and
paired-bootstrap significance test. Does NOT reuse mask_fill_generate
directly since that function has no hook into which positions get masked
before generation runs -- reimplemented here with the identical masking
logic (same seed convention) so the patch can target exactly those positions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l42_run_repro import (
    MASK_FRACTION,
    MODEL_NAME,
    N_EVAL_SEQS,
    N_VECTOR_SEQS_PER_GROUP,
    DATA_PATH,
    MAX_SEQ_LEN,
    SEED,
    mean_pooled_activation_all_layers,
    score_thermostability_proxy,
)
from src.l38.l42_steering_repro import (
    is_degenerate_sequence,
    paired_bootstrap_mean_diff,
    split_by_percentile,
)

OUT_PATH = Path(__file__).resolve().parent / "l47_task_b_out.json"
N_BOOT = 10000
MIN_NONDEGENERATE_PAIRS = 30


class MaskedPositionPatchHook:
    """Replaces a layer's residual-stream output with a fixed mean-pooled
    vector, ONLY at the given mask_positions -- every other position keeps
    its own real activation untouched. Distinct from L42's
    MultiLayerSteeringHook (which ADDS alpha*direction everywhere and
    renormalizes): this is true patching (substitution) at a targeted
    subset of positions, not a global perturbation.
    """

    def __init__(self, patch_vector: torch.Tensor, mask_positions: torch.Tensor):
        self.patch_vector = patch_vector
        self.mask_positions = mask_positions

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        current = output[0] if is_tuple else output
        patched = current.clone()
        patch_vec = self.patch_vector.to(current.device, current.dtype)
        patched[:, self.mask_positions, :] = patch_vec
        if is_tuple:
            return (patched,) + output[1:]
        return patched


def get_mask_positions(tokenizer, sequence: str, mask_fraction: float, seed: int, max_len: int):
    """Identical masking logic to l42_run_repro.mask_fill_generate, factored
    out so the mask positions can be computed BEFORE registering the patch
    hook (the hook needs to know which positions to target)."""
    seq = sequence[:max_len]
    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2)
    input_ids = enc["input_ids"][0].clone()

    rng = torch.Generator().manual_seed(seed)
    special_ids = set(tokenizer.all_special_ids)
    non_special_positions = torch.tensor([i for i, t in enumerate(input_ids.tolist()) if t not in special_ids])
    n_mask = max(1, int(len(non_special_positions) * mask_fraction))
    perm = torch.randperm(len(non_special_positions), generator=rng)
    mask_positions = non_special_positions[perm[:n_mask]]
    return input_ids, mask_positions, enc["attention_mask"][0]


@torch.no_grad()
def mask_fill_generate_with_patch(model, tokenizer, sequence, mask_fraction, seed, device, patch_vector=None, patch_layer=None, max_len=MAX_SEQ_LEN):
    """Same single-shot masked-marginal generation as L42's
    mask_fill_generate, but if patch_vector/patch_layer are given, registers
    a MaskedPositionPatchHook on that layer for the duration of this forward
    pass, restricted to the SAME positions being masked/predicted."""
    input_ids, mask_positions, attention_mask = get_mask_positions(tokenizer, sequence, mask_fraction, seed, max_len)

    masked_ids = input_ids.clone()
    masked_ids[mask_positions] = tokenizer.mask_token_id

    handle = None
    if patch_vector is not None:
        hook = MaskedPositionPatchHook(patch_vector, mask_positions)
        handle = model.esm.encoder.layer[patch_layer].register_forward_hook(hook)

    try:
        masked_enc = {"input_ids": masked_ids.unsqueeze(0).to(device), "attention_mask": attention_mask.unsqueeze(0).to(device)}
        out = model(**masked_enc)
    finally:
        if handle is not None:
            handle.remove()

    predicted_ids = out.logits.argmax(dim=-1).squeeze(0).cpu()
    filled_ids = masked_ids.clone()
    filled_ids[mask_positions] = predicted_ids[mask_positions]

    tokens_str = tokenizer.convert_ids_to_tokens(filled_ids.tolist())
    return "".join(t for t in tokens_str if t not in tokenizer.all_special_tokens)


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
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
    print(f"eval sequences: {len(eval_sequences)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"model loaded: {MODEL_NAME}, {n_layers} layers", flush=True)

    print("\nembedding low-Tm group (all layers)...", flush=True)
    low_activations = mean_pooled_activation_all_layers(model, tokenizer, low_group, device)
    print("embedding high-Tm group (all layers)...", flush=True)
    high_activations = mean_pooled_activation_all_layers(model, tokenizer, high_group, device)

    high_patch_vectors = {
        layer: torch.tensor(high_activations[layer].mean(axis=0), dtype=torch.float32, device=device)
        for layer in range(n_layers)
    }
    low_patch_vectors = {
        # control: patch with the LOW-Tm group's own mean -- if this produces
        # the same effect as the high-Tm patch, the effect isn't specific to
        # the high-Tm signal, it's just "replace with any fixed vector."
        layer: torch.tensor(low_activations[layer].mean(axis=0), dtype=torch.float32, device=device)
        for layer in range(n_layers)
    }

    def generate_with_patch(patch_vectors, layer):
        generated = [
            mask_fill_generate_with_patch(
                model, tokenizer, seq, MASK_FRACTION, SEED + i, device,
                patch_vector=patch_vectors[layer], patch_layer=layer,
            )
            for i, seq in enumerate(eval_sequences)
        ]
        scores = score_thermostability_proxy(generated)
        return generated, scores

    print("\n=== baseline (no patch) ===", flush=True)
    baseline_generated = [
        mask_fill_generate_with_patch(model, tokenizer, seq, MASK_FRACTION, SEED + i, device)
        for i, seq in enumerate(eval_sequences)
    ]
    baseline_scores = score_thermostability_proxy(baseline_generated)
    baseline_degenerate = np.array([is_degenerate_sequence(s) for s in baseline_generated])
    print(f"baseline mean: {baseline_scores.mean():.4f}, degenerate: {baseline_degenerate.sum()}/{len(baseline_degenerate)}", flush=True)

    results = {"baseline_mean": float(baseline_scores.mean()), "by_layer": {}}

    for layer in range(n_layers):
        high_generated, high_scores = generate_with_patch(high_patch_vectors, layer)
        low_generated, low_scores = generate_with_patch(low_patch_vectors, layer)

        high_degenerate = np.array([is_degenerate_sequence(s) for s in high_generated])
        low_degenerate = np.array([is_degenerate_sequence(s) for s in low_generated])
        keep = ~high_degenerate & ~low_degenerate & ~baseline_degenerate
        n_kept = int(keep.sum())

        entry = {
            "high_patch_mean": float(high_scores.mean()), "low_patch_mean": float(low_scores.mean()),
            "n_degenerate_high": int(high_degenerate.sum()), "n_degenerate_low": int(low_degenerate.sum()),
        }
        if n_kept < MIN_NONDEGENERATE_PAIRS:
            entry.update({
                "point_estimate": None, "ci_lower": None, "ci_upper": None, "significant_at_95pct": False,
                "n": n_kept, "excluded_reason": f"only {n_kept} non-degenerate pairs",
            })
        else:
            bootstrap = paired_bootstrap_mean_diff(low_scores[keep], high_scores[keep], n_boot=N_BOOT, seed=SEED)
            entry.update(bootstrap)

        results["by_layer"][layer] = entry
        sig_marker = " <-- SIGNIFICANT" if entry["significant_at_95pct"] else ""
        print(f"layer {layer:2d}: high_patch={entry['high_patch_mean']:.4f} low_patch={entry['low_patch_mean']:.4f} "
              f"diff={entry.get('point_estimate')} n={entry['n']}{sig_marker}", flush=True)

    significant_layers = [l for l, e in results["by_layer"].items() if e["significant_at_95pct"]]
    results["significant_layers"] = significant_layers
    print(f"\n=== LAYERS WHERE HIGH-Tm PATCH SIGNIFICANTLY BEATS LOW-Tm PATCH ===", flush=True)
    print(significant_layers, flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
