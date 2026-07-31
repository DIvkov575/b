"""L46: run InterPLM's pretrained SAE over real sequences and rank discovered
features by selectivity, with ZERO target property specified anywhere in
this process -- the opposite of L42/L43/L45, which all required choosing a
property (thermostability, solubility) before running anything.

Method: encode many real sequences' per-residue activations through the
SAE, track every feature's firing pattern, then rank features by a
selectivity score that doesn't reference any biological label: how
sparse/concentrated a feature's firing is (fires strongly on a small
fraction of residues, rather than diffusely everywhere -- InterPLM's own
finding is that the useful, interpretable features look like this, while
uninteresting/noise-fitting features fire diffusely). This is a real
proxy for "found a genuine pattern" vs. "found nothing," computable BEFORE
any human or downstream label ever gets involved.

After ranking, the TOP features get characterized post-hoc (which
sequences/residues each one fires most strongly on) purely for a human
to read and interpret -- this final labeling step is unavoidable (a
u ndiscovered feature has no name until someone looks at what it
responds to), but the discovery/ranking step itself is fully unsupervised.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.l38.l46_sae_feature_discovery import (
    ESM_DIM,
    FEATURE_DIM,
    MODEL_NAME,
    extract_per_residue_activations,
    load_interplm_sae,
)

DATA_PATH = Path(__file__).resolve().parent / "data_cache" / "meltome" / "mixed_split.csv"
OUT_PATH = Path(__file__).resolve().parent / "l46_discovery_out.json"

LAYER = 24  # one of L45's causally-significant layers for thermostability --
# chosen so a later cross-check ("does the unsupervised discovery at this
# layer overlap with what the supervised sweep found causally relevant
# here") is possible, without that comparison influencing this discovery
# step itself (this script never touches a Tm label).
N_SEQUENCES = 100
MAX_SEQ_LEN = 400
SEED = 0
TOP_K_FEATURES = 15


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df["sequence"].str.len() <= MAX_SEQ_LEN].reset_index(drop=True)
    sequences = df["sequence"].sample(n=N_SEQUENCES, random_state=SEED).tolist()
    print(f"sampled {len(sequences)} real sequences (labels never loaded/used)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME).to(device).eval()
    sae = load_interplm_sae(layer=LAYER, device=device)
    print(f"SAE loaded for layer {LAYER}", flush=True)

    # For each feature: track max activation seen, and WHERE (which sequence
    # index + residue position + the residue identity itself) it fired
    # strongest -- this is what lets us characterize a feature post-hoc
    # without ever having specified what to look for.
    max_activation = np.zeros(FEATURE_DIM)
    max_activation_location = [None] * FEATURE_DIM  # (seq_idx, residue_idx, residue_char)
    total_activation = np.zeros(FEATURE_DIM)
    total_active_count = np.zeros(FEATURE_DIM)  # how many (seq, residue) pairs activate this feature at all
    total_residues = 0

    per_seq_tokens = []

    for seq_idx, seq in enumerate(sequences):
        hidden, tokens = extract_per_residue_activations(model, tokenizer, seq, layer=LAYER, device=device)
        residue_tokens = tokens[1:-1]  # drop <cls>, <eos>
        residue_hidden = hidden[1:-1]
        per_seq_tokens.append(residue_tokens)

        with torch.no_grad():
            features = sae.encode(residue_hidden).cpu().numpy()  # [n_residues, FEATURE_DIM]

        total_residues += features.shape[0]
        total_activation += features.sum(axis=0)
        total_active_count += (features > 0).sum(axis=0)

        seq_max = features.max(axis=0)
        improved = seq_max > max_activation
        if improved.any():
            argmax_residue = features.argmax(axis=0)
            for feat_idx in np.where(improved)[0]:
                residue_idx = int(argmax_residue[feat_idx])
                max_activation_location[feat_idx] = (seq_idx, residue_idx, residue_tokens[residue_idx])
            max_activation = np.maximum(max_activation, seq_max)

        if (seq_idx + 1) % 20 == 0:
            print(f"  processed {seq_idx + 1}/{len(sequences)} sequences", flush=True)

    mean_activation_when_active = np.divide(
        total_activation, total_active_count, out=np.zeros_like(total_activation), where=total_active_count > 0
    )
    firing_fraction = total_active_count / total_residues  # what fraction of ALL residues activate this feature at all

    # Selectivity score: high max activation AND low firing fraction --
    # a feature that fires very strongly but RARELY is a candidate for a
    # specific, localized pattern; a feature that fires everywhere
    # (firing_fraction near 1) or never (max_activation near 0) is not
    # interesting regardless of any label. This mirrors InterPLM's own
    # observation that useful features are sparse/localized, without
    # requiring any biological annotation to compute.
    with np.errstate(divide="ignore"):
        selectivity = max_activation * (1.0 - firing_fraction)

    # Exclude dead features (never fired at all -- not selective, just unused)
    dead_mask = max_activation == 0
    selectivity[dead_mask] = -np.inf

    top_feature_ids = np.argsort(-selectivity)[:TOP_K_FEATURES]

    results = {
        "layer": LAYER,
        "n_sequences": N_SEQUENCES,
        "n_dead_features": int(dead_mask.sum()),
        "n_total_features": FEATURE_DIM,
        "top_features": [],
    }

    print(f"\n=== TOP {TOP_K_FEATURES} MOST SELECTIVE FEATURES (layer {LAYER}) ===", flush=True)
    for feat_id in top_feature_ids:
        loc = max_activation_location[feat_id]
        seq_idx, residue_idx, residue_char = loc
        context_start = max(0, residue_idx - 5)
        context_end = min(len(per_seq_tokens[seq_idx]), residue_idx + 6)
        context = "".join(per_seq_tokens[seq_idx][context_start:context_end])
        marked_context = (
            "".join(per_seq_tokens[seq_idx][context_start:residue_idx])
            + f"[{residue_char}]"
            + "".join(per_seq_tokens[seq_idx][residue_idx + 1:context_end])
        )
        entry = {
            "feature_id": int(feat_id),
            "max_activation": float(max_activation[feat_id]),
            "firing_fraction": float(firing_fraction[feat_id]),
            "mean_activation_when_active": float(mean_activation_when_active[feat_id]),
            "selectivity_score": float(selectivity[feat_id]),
            "strongest_firing_residue": residue_char,
            "strongest_firing_context": marked_context,
        }
        results["top_features"].append(entry)
        print(f"feature {feat_id}: max_act={entry['max_activation']:.2f} "
              f"fires_on={entry['firing_fraction']*100:.3f}% of residues, "
              f"strongest at [{residue_char}] in ...{marked_context}...", flush=True)

    print(f"\ndead features (never fired across {total_residues} residues): {dead_mask.sum()}/{FEATURE_DIM}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
