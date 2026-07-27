"""L39 v3 -- real PVP/non-PVP classification eval at ESM-2-650M scale.

Reuses virion_eval.py's bootstrap-tested comparison machinery (paired
bootstrap significance, not just point estimates), pointed at the real
PhaVIP-lineage dataset (phavip_real_data.py) and the 650M base/fine-tuned
checkpoints, instead of the original 35M-scale UniProt-keyword task.

Run: .venv-l38/bin/python -m src.l38.run_phavip_650m_eval
"""
import json
from pathlib import Path

import torch

from src.l38.phavip_real_data import load_pvp_labeled_dataset
from src.l38.train_phavip_650m import MAX_LENGTH, MODEL_NAME, OUT_DIR
from src.l38.virion_eval import run_comparison

RESULTS_PATH = Path(__file__).resolve().parent / "phavip_650m_classification_results.json"

# Real 650M embeddings at length 700 are heavier than the 35M run's --
# smaller batch to stay well under the A10G's 23GB, matching the training
# script's own OOM lesson (see train_phavip_650m.py's BATCH_SIZE comment).
EMBED_BATCH_SIZE = 4


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    finetuned_model_path = str(OUT_DIR / "final_model")

    results = run_comparison(
        base_model_path=MODEL_NAME,
        finetuned_model_path=finetuned_model_path,
        device=device,
        load_data_fn=load_pvp_labeled_dataset,
        max_length=MAX_LENGTH,
        embed_batch_size=EMBED_BATCH_SIZE,
    )

    # Strip the large per-sequence arrays before saving, matching the
    # convention already used for the 35M-scale results (see
    # virion_eval_results_paired.json) -- the aggregate stats are what
    # matters for the write-up, not 65K raw predictions.
    for model_key in ["base", "finetuned"]:
        for field in ["y_test", "preds", "probs"]:
            results[model_key].pop(field, None)

    print("\n=== FINAL (stripped) RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
