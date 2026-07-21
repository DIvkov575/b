"""L41 Gate 2: causal steering via forward hook + iterative mask-fill generation.

Takes held-out NON-kinase sequences, masks a fraction of positions, and
regenerates them with (a) no steering, (b) the Gate-1 winning feature's
decoder direction added at layer 20, (c) a random direction of matched norm
(the pre-registered control -- docs/L41_PROTOCOL.md Gate 2). Saves all
generated sequences for Gate 3's independent classifier evaluation.
"""
import json
from pathlib import Path

import numpy as np
import torch

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

GATE1_OUT = Path(__file__).resolve().parent / "l41_gate1_out"
OUT_DIR = Path(__file__).resolve().parent / "l41_gate2_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 20
MASK_FRACTION = 0.8  # v2: bumped from 0.3 -- Gate 3 v1 found the classifier's
# whole-sequence score was dominated by the ~70% UNMASKED residues, making it
# structurally unable to detect any local steering effect regardless of
# whether steering worked (both P(kinase) sat at ~0.003 for every condition,
# baseline included). Masking most of the sequence gives steering room to
# actually move the classifier's verdict. This is a measurement-sensitivity
# fix, not a change to the PASS/KILL decision rule itself.
ALPHAS = [0.0, 5.0, 10.0, 20.0]  # 0.0 = unsteered baseline
N_SEQS_TO_STEER = 60  # subset of the held-out non-kinase eval split
SEED = 0
MAX_SEQ_LEN = 400  # keep per-sequence forward passes cheap during the mask-fill loop


class SteeringHook:
    """Adds `alpha * direction` to the residual-stream output of the hooked
    transformer block, at every position, every forward pass while active."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = direction
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden + self.alpha * self.direction
            return (hidden,) + output[1:]
        return output + self.alpha * self.direction


@torch.no_grad()
def mask_fill_generate(model, tokenizer, sequence: str, mask_fraction: float, seed: int, device: str) -> str:
    """Single-pass mask-fill: mask `mask_fraction` of positions (deterministic
    given seed), predict the highest-probability residue for each masked
    position in one forward pass (not iterative refinement -- a single-shot
    approximation, sufficient for this Gate 2 check; refinement is a natural
    extension if this passes)."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    tokens = protein_tensor.sequence.clone().cpu()

    rng = torch.Generator().manual_seed(seed)
    non_special_positions = torch.arange(1, len(tokens) - 1)  # skip BOS/EOS
    n_mask = max(1, int(len(non_special_positions) * mask_fraction))
    perm = torch.randperm(len(non_special_positions), generator=rng)
    mask_positions = non_special_positions[perm[:n_mask]]

    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = tokenizer.mask_token_id

    from esm.sdk.api import ESMProteinTensor
    masked_protein_tensor = ESMProteinTensor(sequence=masked_tokens.to(device))

    output = model.logits(masked_protein_tensor, LogitsConfig(sequence=True))
    predicted_ids = output.logits.sequence.argmax(dim=-1).squeeze(0).cpu()

    filled_tokens = masked_tokens.clone()
    filled_tokens[mask_positions] = predicted_ids[mask_positions]

    tokens_str = tokenizer.convert_ids_to_tokens(filled_tokens.tolist())
    decoded = "".join(t for t in tokens_str if t not in tokenizer.all_special_tokens)
    return decoded


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    with open(GATE1_OUT / "eval_split.json") as f:
        eval_split = json.load(f)
    non_kinase_eval = eval_split["non_kinase_eval"]
    non_kinase_eval = [s for s in non_kinase_eval if len(s) <= MAX_SEQ_LEN][:N_SEQS_TO_STEER]
    print(f"steering {len(non_kinase_eval)} held-out non-kinase sequences", flush=True)

    steering_vector_np = np.load(GATE1_OUT / "steering_vector.npy")
    steering_vector = torch.tensor(steering_vector_np, dtype=torch.float32, device=device)
    steering_vector = steering_vector / steering_vector.norm()  # unit direction; alpha controls magnitude

    rng = np.random.RandomState(SEED)
    random_direction_np = rng.normal(size=steering_vector_np.shape)
    random_direction = torch.tensor(random_direction_np, dtype=torch.float32, device=device)
    random_direction = random_direction / random_direction.norm()

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    from esm.tokenization import get_esmc_model_tokenizers
    tokenizer = get_esmc_model_tokenizers()

    results = {"unsteered": {}, "real_direction": {}, "random_control": {}}

    print("\n=== Condition: unsteered (alpha=0) ===", flush=True)
    results["unsteered"][0.0] = [
        mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED, device) for seq in non_kinase_eval
    ]
    print(f"  generated {len(results['unsteered'][0.0])} sequences", flush=True)

    for alpha in ALPHAS:
        if alpha == 0.0:
            continue
        print(f"\n=== Condition: real_direction, alpha={alpha} ===", flush=True)
        hook_handle = model.transformer.blocks[LAYER].register_forward_hook(
            SteeringHook(steering_vector, alpha)
        )
        try:
            results["real_direction"][alpha] = [
                mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED, device) for seq in non_kinase_eval
            ]
        finally:
            hook_handle.remove()
        print(f"  generated {len(results['real_direction'][alpha])} sequences", flush=True)

        print(f"\n=== Condition: random_control, alpha={alpha} ===", flush=True)
        hook_handle = model.transformer.blocks[LAYER].register_forward_hook(
            SteeringHook(random_direction, alpha)
        )
        try:
            results["random_control"][alpha] = [
                mask_fill_generate(model, tokenizer, seq, MASK_FRACTION, SEED, device) for seq in non_kinase_eval
            ]
        finally:
            hook_handle.remove()
        print(f"  generated {len(results['random_control'][alpha])} sequences", flush=True)

    with open(OUT_DIR / "generated_sequences.json", "w") as f:
        json.dump({"original": non_kinase_eval, "generated": results}, f, indent=2)
    print(f"\nSaved all generated sequences to {OUT_DIR / 'generated_sequences.json'}", flush=True)


if __name__ == "__main__":
    main()
