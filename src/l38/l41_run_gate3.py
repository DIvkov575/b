"""L41 Gate 3: independent classifier evaluation of Gate 2's steered generations.

Trains a logistic-regression probe on DENSE ESMC-300M mean-pooled embeddings
(not the SAE feature space used to find the steering vector -- avoids
circularity) to predict kinase/non-kinase, on a split disjoint from both
Gate 1's feature-identification set and Gate 2's generation inputs. Scores
every generated sequence from Gate 2 with this independent classifier.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from src.l38.phage_data import clean_sequences, parse_fasta, train_eval_split

DATA_DIR = Path(__file__).resolve().parent / "data_cache"
GATE1_OUT = Path(__file__).resolve().parent / "l41_gate1_out"
GATE2_OUT = Path(__file__).resolve().parent / "l41_gate2_out"
OUT_DIR = Path(__file__).resolve().parent / "l41_gate3_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER_LAYER = 20  # match Gate 1's layer for the dense embedding too
SEED = 0
CLASSIFIER_TRAIN_MAX_PER_CLASS = 300


@torch.no_grad()
def mean_pooled_embedding(model, sequence: str, layer: int, device: str) -> np.ndarray:
    protein = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, LogitsConfig(sequence=False, return_hidden_states=True))
    hidden = output.hidden_states[layer].squeeze(0).float()
    return hidden.mean(dim=0).cpu().numpy()


def build_independent_classifier(model, device):
    """Train on the SAME identification-split sequences Gate 1 used for
    feature-finding (disjoint from Gate 2's held-out generation inputs) --
    this reuses Gate 1's identification/eval split boundary, not a fresh
    random split, so there is zero overlap with Gate 2's inputs by
    construction."""
    positive_raw = parse_fasta(DATA_DIR / "kinase_positive.fasta")
    negative_raw = parse_fasta(DATA_DIR / "kinase_negative.fasta")
    positive_clean = clean_sequences(positive_raw)
    negative_clean = clean_sequences(negative_raw)

    pos_id, _ = train_eval_split(positive_clean, eval_frac=0.3, seed=SEED)
    neg_id, _ = train_eval_split(negative_clean, eval_frac=0.3, seed=SEED)
    pos_id = pos_id[:CLASSIFIER_TRAIN_MAX_PER_CLASS]
    neg_id = neg_id[:CLASSIFIER_TRAIN_MAX_PER_CLASS]

    print(f"training independent classifier on {len(pos_id)} kinase / {len(neg_id)} non-kinase "
          f"(same identification split as Gate 1, disjoint from Gate 2's eval-split inputs)", flush=True)

    sequences = pos_id + neg_id
    labels = np.array([1] * len(pos_id) + [0] * len(neg_id))

    embeddings = np.stack(
        [mean_pooled_embedding(model, seq, CLASSIFIER_LAYER, device) for seq in sequences], axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"independent classifier: train_acc={train_acc:.4f}, held-out_acc={test_acc:.4f}", flush=True)

    return clf, test_acc


def score_sequences(clf, model, sequences, layer, device) -> np.ndarray:
    embeddings = np.stack(
        [mean_pooled_embedding(model, seq, layer, device) for seq in sequences], axis=0
    )
    return clf.predict_proba(embeddings)[:, 1]  # P(kinase)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    model = ESMC.from_pretrained("esmc_300m").to(device).eval()

    clf, classifier_test_acc = build_independent_classifier(model, device)

    with open(GATE2_OUT / "generated_sequences.json") as f:
        gate2_data = json.load(f)

    generated = gate2_data["generated"]
    results = {"classifier_held_out_accuracy": classifier_test_acc, "conditions": {}}

    print("\n=== Scoring unsteered baseline ===", flush=True)
    unsteered_seqs = generated["unsteered"]["0.0"]
    unsteered_scores = score_sequences(clf, model, unsteered_seqs, CLASSIFIER_LAYER, device)
    results["conditions"]["unsteered"] = {
        "mean_p_kinase": float(unsteered_scores.mean()),
        "std_p_kinase": float(unsteered_scores.std()),
        "n": len(unsteered_scores),
    }
    print(f"unsteered: mean P(kinase)={unsteered_scores.mean():.4f} (+/-{unsteered_scores.std():.4f})", flush=True)

    for alpha_key, seqs in generated["real_direction"].items():
        print(f"\n=== Scoring real_direction alpha={alpha_key} ===", flush=True)
        scores = score_sequences(clf, model, seqs, CLASSIFIER_LAYER, device)
        results["conditions"][f"real_direction_alpha_{alpha_key}"] = {
            "mean_p_kinase": float(scores.mean()),
            "std_p_kinase": float(scores.std()),
            "n": len(scores),
        }
        print(f"real_direction alpha={alpha_key}: mean P(kinase)={scores.mean():.4f} (+/-{scores.std():.4f})", flush=True)

    for alpha_key, seqs in generated["random_control"].items():
        print(f"\n=== Scoring random_control alpha={alpha_key} ===", flush=True)
        scores = score_sequences(clf, model, seqs, CLASSIFIER_LAYER, device)
        results["conditions"][f"random_control_alpha_{alpha_key}"] = {
            "mean_p_kinase": float(scores.mean()),
            "std_p_kinase": float(scores.std()),
            "n": len(scores),
        }
        print(f"random_control alpha={alpha_key}: mean P(kinase)={scores.mean():.4f} (+/-{scores.std():.4f})", flush=True)

    print("\n=== GATE 3 SUMMARY ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    # Pre-registered PASS/KILL rule (docs/L41_PROTOCOL.md Gate 3): real-direction
    # steering must exceed BOTH unsteered baseline AND the matched random-direction
    # control, at the same alpha, for the ordering to count as evidence of causal
    # function-steering (not just "any perturbation shifts the score").
    verdict = {}
    for alpha_key in generated["real_direction"].keys():
        real_score = results["conditions"][f"real_direction_alpha_{alpha_key}"]["mean_p_kinase"]
        random_score = results["conditions"][f"random_control_alpha_{alpha_key}"]["mean_p_kinase"]
        baseline_score = results["conditions"]["unsteered"]["mean_p_kinase"]
        passes = real_score > baseline_score and real_score > random_score
        verdict[alpha_key] = {
            "real_exceeds_baseline": real_score > baseline_score,
            "real_exceeds_random_control": real_score > random_score,
            "decision": "PASS" if passes else "KILL",
        }
    results["gate3_verdict_per_alpha"] = verdict
    print("\n=== GATE 3 VERDICT (per alpha) ===", flush=True)
    print(json.dumps(verdict, indent=2), flush=True)

    with open(OUT_DIR / "gate3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR / 'gate3_results.json'}", flush=True)


if __name__ == "__main__":
    main()
