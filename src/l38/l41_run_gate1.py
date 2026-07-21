"""L41 Gate 1 runner: find a kinase-linked SAE feature in ESMC-300M.

Requires ESMC-300M + its SAE checkpoints loaded on GPU -- not unit-testable
without one; the pure math (cohens_d, sae_encode/decode, gate1_decision)
is tested in tests/l38/test_l41_steering.py against synthetic data.
"""
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from src.l38.l41_steering import gate1_decision, rank_features_by_separation, sae_encode
from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split

DATA_DIR = Path(__file__).resolve().parent / "data_cache"
SAE_DIR = DATA_DIR / "esmc_sae"
OUT_DIR = Path(__file__).resolve().parent / "l41_gate1_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = 20
K = 64
D_MODEL = 960
COHENS_D_THRESHOLD = 1.0
SEED = 0
MAX_SEQS_PER_CLASS = 300  # keep Gate 1 identification cheap; eval split reserved for Gate 3


def load_sae(layer: int, device: str):
    path = SAE_DIR / f"layer_{layer}.safetensors"
    tensors = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key).to(device)
    return tensors["W_enc"], tensors["W_dec"], tensors["b_dec"]


@torch.no_grad()
def mean_pooled_layer_activation(model, sequences, layer, device, batch_log_every=50):
    """Returns [n_seqs, d_model] mean-pooled hidden state at `layer` for each sequence.
    ESMC's SDK processes one ESMProteinTensor at a time (no batched forward in
    the public API used here), so this loops -- acceptable at the ~300-seq
    Gate 1 identification scale."""
    activations = []
    for i, seq in enumerate(sequences):
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        output = model.logits(
            protein_tensor,
            LogitsConfig(sequence=False, return_hidden_states=True),
        )
        hidden = output.hidden_states[layer].squeeze(0).float()  # [seq_len, d_model]
        activations.append(hidden.mean(dim=0).cpu().numpy())
        if (i + 1) % batch_log_every == 0:
            print(f"  embedded {i+1}/{len(sequences)}", flush=True)
    return np.stack(activations, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    positive_raw = parse_fasta(DATA_DIR / "kinase_positive.fasta")
    negative_raw = parse_fasta(DATA_DIR / "kinase_negative.fasta")
    positive_clean = clean_sequences(positive_raw)
    negative_clean = clean_sequences(negative_raw)
    print(f"kinase positive (clean): {len(positive_clean)}, negative (clean): {len(negative_clean)}", flush=True)

    # Split: identification split (used here, Gate 1) vs eval split (reserved
    # for Gate 2 generation inputs + Gate 3 independent classifier -- disjoint
    # per docs/L41_PROTOCOL.md leakage discipline).
    pos_id, pos_eval = train_eval_split(positive_clean, eval_frac=0.3, seed=SEED)
    neg_id, neg_eval = train_eval_split(negative_clean, eval_frac=0.3, seed=SEED)

    pos_id = pos_id[:MAX_SEQS_PER_CLASS]
    neg_id = neg_id[:MAX_SEQS_PER_CLASS]
    print(f"identification split: {len(pos_id)} positive, {len(neg_id)} negative", flush=True)

    # Persist the eval splits now so Gate 2/3 use the exact same disjoint sets.
    with open(OUT_DIR / "eval_split.json", "w") as f:
        json.dump({"kinase_eval": pos_eval, "non_kinase_eval": neg_eval}, f)

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()
    W_enc, W_dec, b_dec = load_sae(LAYER, device)
    W_enc_np, b_dec_np = W_enc.cpu().numpy(), b_dec.cpu().numpy()

    print(f"\nembedding positive (kinase) sequences at layer {LAYER}...", flush=True)
    pos_activations = mean_pooled_layer_activation(model, pos_id, LAYER, device)
    print(f"embedding negative (non-kinase) sequences at layer {LAYER}...", flush=True)
    neg_activations = mean_pooled_layer_activation(model, neg_id, LAYER, device)

    pos_features = sae_encode(pos_activations, W_enc_np, b_dec_np, k=K)
    neg_features = sae_encode(neg_activations, W_enc_np, b_dec_np, k=K)

    ranked = rank_features_by_separation(pos_features, neg_features)
    decision = gate1_decision(ranked, threshold=COHENS_D_THRESHOLD)

    print(f"\ntop 10 features by |Cohen's d|:", flush=True)
    for idx, d in ranked[:10]:
        print(f"  feature {idx}: d={d:.4f}", flush=True)

    print(f"\n=== GATE 1 DECISION ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)

    results = {
        "layer": LAYER,
        "k": K,
        "n_positive_id": len(pos_id),
        "n_negative_id": len(neg_id),
        "top_10_features": ranked[:10],
        "decision": decision,
    }
    with open(OUT_DIR / "gate1_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if decision["decision"] == "PASS":
        winning_feature = decision["winning_feature"]
        steering_vector = W_dec[winning_feature].cpu().numpy()
        np.save(OUT_DIR / "steering_vector.npy", steering_vector)
        print(f"\nSaved steering vector for feature {winning_feature} to {OUT_DIR / 'steering_vector.npy'}", flush=True)

    print(f"\nSaved full results to {OUT_DIR / 'gate1_results.json'}", flush=True)


if __name__ == "__main__":
    main()
