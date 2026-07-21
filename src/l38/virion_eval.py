"""L39 real eval: phage virion-protein classification (frozen embeddings + linear probe).

Reproduces the ESM-PVP (Li & Liang 2023) task shape -- classify whether a
phage protein is a virion/structural protein (capsid, tail, baseplate, etc.)
from sequence alone -- using UniProt's own KW-0946 (Virion) keyword as the
label, sourced directly via UniProt REST (no external repo). This is the
actual downstream benchmark the L39 perplexity result needs to translate
into before it counts as a real win; perplexity alone is not a publishable
result.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from src.l38.phage_data import clean_sequences, parse_fasta

DATA_DIR = Path(__file__).resolve().parent / "data_cache" / "phage"
SEED = 0
MAX_LENGTH = 512
EMBED_BATCH_SIZE = 16


def load_labeled_dataset() -> Tuple[List[str], np.ndarray]:
    positive = clean_sequences(parse_fasta(DATA_DIR / "virion_positive.fasta"))
    negative = clean_sequences(parse_fasta(DATA_DIR / "virion_negative.fasta"))

    # De-duplicate: a sequence appearing in both sets (shouldn't per the KW-0946
    # filter, but cheap to guard against) is dropped from both to avoid a
    # contradictory label.
    pos_set, neg_set = set(positive), set(negative)
    overlap = pos_set & neg_set
    positive = [s for s in positive if s not in overlap]
    negative = [s for s in negative if s not in overlap]

    sequences = positive + negative
    labels = np.array([1] * len(positive) + [0] * len(negative))
    return sequences, labels


@torch.no_grad()
def embed_sequences(sequences: List[str], model_path: str, device: str) -> np.ndarray:
    """Mean-pooled last-hidden-state embedding per sequence, from a FROZEN
    encoder (base or fine-tuned MLM backbone -- the encoder body is shared
    between AutoModelForMaskedLM and AutoModel for ESM-2, so this loads
    cleanly from either a HF hub id or a local fine-tuned checkpoint dir)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()

    embeddings = []
    for i in range(0, len(sequences), EMBED_BATCH_SIZE):
        batch = sequences[i : i + EMBED_BATCH_SIZE]
        enc = tokenizer(
            batch, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        ).to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_probe(embeddings: np.ndarray, labels: np.ndarray, seed: int = SEED) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "auc": float(roc_auc_score(y_test, probs)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def run_comparison(base_model_path: str, finetuned_model_path: str, device: str) -> dict:
    sequences, labels = load_labeled_dataset()
    print(f"loaded {len(sequences)} sequences ({labels.sum()} virion, {len(labels)-labels.sum()} non-virion)", flush=True)

    print(f"embedding with base model: {base_model_path}", flush=True)
    base_embeddings = embed_sequences(sequences, base_model_path, device)
    base_results = evaluate_probe(base_embeddings, labels)
    print(f"base results: {base_results}", flush=True)

    print(f"embedding with fine-tuned model: {finetuned_model_path}", flush=True)
    ft_embeddings = embed_sequences(sequences, finetuned_model_path, device)
    ft_results = evaluate_probe(ft_embeddings, labels)
    print(f"fine-tuned results: {ft_results}", flush=True)

    return {"base": base_results, "finetuned": ft_results}
